# Local Jev-style decision APIs on DGX Spark

Run native decision checkpoints behind a local `/v1/systemone` endpoint. This repository provides the recipe used in our DGX Spark experiments, prebuilt Linux ARM64 container images, and a comparison of ten checkpoints.

The main walkthrough uses **Cloudflare Clef Flash**. It accepts a text state, questions and explicit answer choices. This is a native decision-head API, not a vLLM chat server. Model weights are downloaded separately and are not included in the image.

## Prebuilt images

Package page: [local-jev-api on GitHub Container Registry](https://github.com/krisitown/local-jev-api/pkgs/container/local-jev-api).

| Image | Purpose |
|---|---|
| `ghcr.io/krisitown/local-jev-api:spark-20261013` | Common tested runtime; Clef Flash tutorial |
| `ghcr.io/krisitown/local-jev-api:spark-omni-20261013` | Clef Omni compatibility runtime, Transformers 5.10.2 and CPU-first loading |

These images target **Linux ARM64 / NVIDIA DGX Spark**. They are not general-purpose x86 or Mac GPU images. They contain NVIDIA's CUDA-enabled PyTorch runtime and are approximately 20 GB unpacked, with substantial shared layers. See [image provenance](image-provenance.json) for the original benchmark and published image identities. Registry manifest digests are recorded separately from Docker image IDs.

Prerequisites: a working NVIDIA driver, Docker configured for GPU access, Python 3, curl, Git, and enough free disk and unified RAM. Flash weights occupy about 18 GiB. In our workload its approximate peak host available-memory drop was 36 GiB, greater than its 17.75 GiB loaded CUDA tensor allocation; leave headroom. The launcher never stops other workloads for you.

## Quick start

```sh
# Download the recipe.
git clone https://github.com/krisitown/local-jev-api.git
cd local-jev-api

# Pull the runtime, download the pinned model, launch and check a real response.
./run.sh
```

One command from an empty working directory:

```sh
git clone https://github.com/krisitown/local-jev-api.git && cd local-jev-api && ./run.sh
```

Already have the checkpoint? Pass the complete snapshot directory at revision `8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3`:

```sh
./run.sh /absolute/path/to/clef-flash
```

The image and model revision are pinned by default. An existing snapshot must match the documented revision; the environment variable alone does not verify its contents. The default downloader fetches that revision explicitly and reuses cached files. The launcher retains timestamped health/response logs under `logs/`, and preserves an unsuccessful container for diagnosis. It refuses to replace an existing container with the same name.

The API binds to `127.0.0.1:18081`. Model loading took about 157 seconds in our benchmark. Initial image and model downloads take additional time. The automated launcher has been checked with simulated Docker/HTTP responses; its underlying runtime and inference path were exercised in the completed experiment. We did not stop an existing model service to rerun the full installer during publication.

## Step by step

```sh
# Pull the prebuilt API runtime.
docker pull ghcr.io/krisitown/local-jev-api:spark-20261013

# Download the complete, pinned Hugging Face snapshot.
python3 ops/download_model.py --key clef-flash

# Start the API with read-only access to the model directory.
docker run -d --name local-jev-flash --gpus all --ipc=host \
  -p 127.0.0.1:18081:18081 \
  -v "$PWD/checkpoints/clef-flash:/model:ro" \
  -e MODEL_PATH=/model \
  -e MODEL_ID=Cloudflare/clef-flash \
  -e MODEL_REVISION=8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3 \
  -e BACKEND=clef \
  ghcr.io/krisitown/local-jev-api:spark-20261013

# Follow loading; Ctrl-C stops the log viewer, not the container.
docker logs -f local-jev-flash

# Once startup completes, check health and send a decision request.
curl --fail http://127.0.0.1:18081/health
curl --fail http://127.0.0.1:18081/v1/systemone \
  -H 'Content-Type: application/json' --data-binary @request.json
```

The example's policy says only idempotent operations may be retried. Choice A (`GET /status`) is the expected policy answer. Inspect the actual result; a smoke test is not an accuracy evaluation.

The wrapper adds `/inspect`, `/metrics` and `_benchmark` telemetry. It rejects oversized input with HTTP 413 instead of silently dropping state text. The 16,384-token Flash limit includes encoded questions and options, not just state. The endpoint is an experimental text API; it has no authentication layer and should remain on loopback unless you add suitable access controls. Downloads access external registries; inference uses the locally mounted snapshot with Hugging Face offline settings.

## Build it yourself

The Dockerfile starts from the exact NVIDIA PyTorch base used in the experiment, installs pinned direct dependencies and copies the API code. NVIDIA's CUDA-enabled PyTorch is retained.

```sh
# Build on the Spark; unnecessary when using our prebuilt image.
docker build -t local-jev-api:spark-local .
```

Use `local-jev-api:spark-local` in the manual `docker run` command. Building today may resolve different transitive dependencies, so it is not guaranteed to reproduce the historical image byte for byte. Pull by a recorded registry digest when exact image identity matters.

For Omni:

```sh
docker build -f Dockerfile.omni -t local-jev-api:spark-omni-local .
```

Omni uses its own checkpoint and `MODEL_ID=Cloudflare/clef-omni`; it is not a drop-in name change in the Flash launcher. The common runtime also contains Laya, Jeff, PPLX and NeoHorse loaders, but their snapshot layouts and native limits differ. See the [checkpoint manifest](references/runtime-recipes/model-pins.json) and [runtime backends](runtime/backends.py). This release's automated launcher supports **Clef Flash only**.

`Dockerfile.publish` is a maintainer recipe: it adds repository labels and license notices to the already benchmarked images while preserving all original runtime layers. It is not the normal from-scratch build path.

## Use from another machine

Keep the API private on the Spark and forward it over SSH:

```sh
ssh -N -L 127.0.0.1:18081:127.0.0.1:18081 USER@SPARK_HOST
```

The local application can then use `http://127.0.0.1:18081/v1/systemone`, model `Cloudflare/clef-flash`, and no bearer key. The [Jev CI adapter](https://github.com/krisitown/jev-quality-gate/tree/local-systemone-adapter) supports a configurable endpoint/model and optional credentials. Its [pinned Flask PR integration](https://github.com/krisitown/jev-quality-gate/blob/c2b608ed0bc35bb05e0bc585c49f6ed3130f7646/validation/public_prs/LOCAL_SYSTEMONE.md) demonstrates transport, typed responses and replay. The overall policy result was uncertain because global reconciliation was not performed; it is not a code-review accuracy result.

## What did we measure?

On one Spark, ten checkpoints received 336 frozen cases each: **3,360 model/case outcomes = 2,065 answers + 1,295 context rejections**. The common 112 short cases support an equal-input comparison. The corpus tests English synthetic retrieval, including missing evidence and linked lookups; it is not a general-purpose leaderboard or a comparison against hosted Jev.

![Shared quality and loaded CUDA allocation](benchmark/figures/quality-memory.svg)

- PPLX: 112/112 shared short cases, about 48.6 GiB loaded CUDA allocation.
- NeoHorse: 104/112 at about 8.55 GiB, but a 2,048-token state limit.
- Flash: 102/112 at about 17.75 GiB, with a 16,384-token serving cap.
- Omni: a successful 63,999-token boundary probe, but only 13/32 correct at the longest scored character tier.

Read [full findings](benchmark/FINDINGS.md) and [machine-readable comparison](benchmark/comparison.json) for speed, memory, context, methodology and limitations. Loaded CUDA memory is not total required RAM. Our Spark has unified memory: do not add GPU allocation and host memory drop. The original interrupted Omni attempt is excluded from the main comparison and documented in the report.

## Stop the demo

```sh
docker stop local-jev-flash
# Remove only the stopped demo container if you want to run the launcher again.
docker rm local-jev-flash
```

Restore any workload you paused before the demo. Host-specific Qwen recovery scripts and credentials are deliberately excluded from this public repository.

## Licenses

Original wrapper/recipe code uses MIT. Third-party runtime libraries retain their notices. NVIDIA container components and separately downloaded model checkpoints retain their own terms; see [NOTICE.md](NOTICE.md). This project is not an official NVIDIA, Cloudflare or Jev product.
