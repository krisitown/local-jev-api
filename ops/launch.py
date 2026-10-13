#!/usr/bin/env python3
"""Download, launch, and verify one of the ten Spark-tested checkpoints."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
PINS = ROOT / "references/runtime-recipes/model-pins.json"
COMMON_IMAGE = "ghcr.io/krisitown/local-jev-api:spark-20261013"
OMNI_IMAGE = "ghcr.io/krisitown/local-jev-api:spark-omni-20261013"
BACKENDS = {
    "clef-flash": "clef", "clef": "clef", "clef-omni": "clef",
    "pplx-decider-v1.1-27b": "pplx", "neohorse-jev-4b": "neohorse",
    "jeff-0.8b": "jeff", "jeff-2b": "jeff",
    "laya": "laya", "laya-multilingual": "laya", "laya-typed-decisions": "laya",
}
REQUIRED = {
    "clef": ("config.json", "joint_schema_model.py", "joint_head.safetensors", "model.safetensors.index.json"),
    "pplx": ("config.json", "decision_config.json", "readout.safetensors", "model.safetensors.index.json", "source/src/autojev/model.py"),
    "neohorse": ("model_manifest.json", "pointer_head.safetensors", "backbone/model.safetensors.index.json"),
    "jeff": ("config.json", "decision_config.json", "readout.safetensors", "tokenizer.json", "model.safetensors"),
    "laya": ("rl_agent_config.json", "model.safetensors", "tokenizer/tokenizer.json"),
}


def fail(message: str) -> None:
    raise RuntimeError(message)


def validate_snapshot(directory: Path, backend: str) -> None:
    if not directory.is_dir():
        fail(f"Checkpoint directory does not exist: {directory}")
    missing = [name for name in REQUIRED[backend] if not (directory / name).is_file()]
    for index in directory.rglob("model.safetensors.index.json"):
        try:
            shards = set(json.loads(index.read_text())["weight_map"].values())
        except (KeyError, ValueError, OSError) as error:
            fail(f"Invalid checkpoint index {index}: {error}")
        missing.extend(str((index.parent / shard).relative_to(directory))
                       for shard in shards if not (index.parent / shard).is_file())
    if missing:
        fail(f"Incomplete {backend} checkpoint at {directory}; missing: {', '.join(sorted(missing))}")


def json_request(url: str, *, payload: dict | None = None, timeout: int = 3) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    request = Request(url, data=data, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def launch(argv: list[str] | None = None) -> int:
    pins = json.loads(PINS.read_text())
    if set(pins) != set(BACKENDS):
        fail("Launcher model keys do not match the pinned checkpoint manifest")
    parser = argparse.ArgumentParser(prog="./run.sh", description=__doc__,
        epilog="With no arguments, launches Clef Flash. IMAGE, CONTAINER_NAME, and PORT may override defaults.")
    parser.add_argument("model", nargs="?", help="Model key, or an existing Clef Flash directory for legacy use")
    parser.add_argument("checkpoint", nargs="?", help="Existing complete checkpoint directory for the selected model")
    parser.add_argument("--list", action="store_true", help="List tested model keys and exit")
    args = parser.parse_args(argv)
    if args.list:
        for key, pin in pins.items():
            print(f"{key:27} {pin['repo']}")
        return 0

    key, supplied = args.model or "clef-flash", args.checkpoint
    if key not in pins and args.model and args.checkpoint is None and Path(args.model).is_dir():
        key, supplied = "clef-flash", args.model
    if key not in pins:
        fail(f"Unknown model {key!r}. Run ./run.sh --list for supported keys.")
    if platform.system() != "Linux" or platform.machine() not in ("aarch64", "arm64"):
        fail("The prebuilt images target Linux ARM64 / DGX Spark.")
    pin, backend = pins[key], BACKENDS[key]
    image = os.environ.get("IMAGE") or (OMNI_IMAGE if key == "clef-omni" else COMMON_IMAGE)
    container = os.environ.get("CONTAINER_NAME") or ("local-jev-flash" if key == "clef-flash" else f"local-jev-{key}")
    port_text = os.environ.get("PORT", "18081")
    if not port_text.isdecimal() or not 1 <= int(port_text) <= 65535:
        fail("PORT must be a number from 1 to 65535")
    port = int(port_text)
    if not container or not all(char.isalnum() or char in "_.-" for char in container):
        fail("CONTAINER_NAME contains invalid characters")
    if subprocess.run(["docker", "container", "inspect", container],
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
        fail(f"Container {container} already exists; inspect it or choose another CONTAINER_NAME.")

    # Cached and locally built images work without a registry round trip.
    if subprocess.run(["docker", "image", "inspect", image],
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0:
        subprocess.run(["docker", "pull", image], check=True)
    if supplied:
        directory = Path(supplied).expanduser().resolve()
    else:
        subprocess.run([sys.executable, str(ROOT / "ops/download_model.py"),
                        "--key", key, "--image", image], check=True)
        directory = ROOT / "checkpoints" / key
    validate_snapshot(directory, backend)

    run = ["docker", "run", "-d", "--name", container, "--gpus", "all", "--ipc=host",
           "-p", f"127.0.0.1:{port}:18081", "-v", f"{directory}:/model:ro",
           "-e", "MODEL_PATH=/model", "-e", f"MODEL_ID={pin['repo']}",
           "-e", f"MODEL_REVISION={pin['revision']}", "-e", f"BACKEND={backend}"]
    if key == "clef-omni":
        run += ["-e", "CLEF_LOAD_DEVICE=cpu"]
    subprocess.run([*run, image], check=True)

    logs = ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log_dir = Path(tempfile.mkdtemp(prefix=f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}-", dir=logs))
    base = f"http://127.0.0.1:{port}"
    expected = {"model": pin["repo"], "revision": pin["revision"], "backend": backend}
    for _ in range(180):
        state = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", container],
                               capture_output=True, text=True, check=True).stdout.strip()
        if state != "true":
            save_logs(container, log_dir)
            fail(f"Container exited. See {log_dir / 'server.log'}")
        try:
            health = json_request(base + "/health")
            (log_dir / "health.json").write_text(json.dumps(health, indent=2) + "\n")
            if isinstance(health, dict) and health.get("ready") is True and all(health.get(k) == v for k, v in expected.items()):
                break
            if isinstance(health, dict) and health.get("ready") is True:
                save_logs(container, log_dir)
                fail(f"Ready container identifies a different model, revision or backend. See {log_dir / 'health.json'} and server.log; container retained.")
        except (HTTPError, URLError, TimeoutError, ValueError, OSError):
            pass
        time.sleep(5)
    else:
        save_logs(container, log_dir)
        fail(f"Readiness timed out or returned another model/revision/backend. See {log_dir / 'server.log'}; container retained.")

    request = json.loads((ROOT / "request.json").read_text())
    request["model"] = pin["repo"]
    try:
        response = json_request(base + "/v1/systemone", payload=request, timeout=120)
    except (HTTPError, URLError, TimeoutError, ValueError, OSError) as error:
        save_logs(container, log_dir)
        fail(f"Decision request failed: {error}. See {log_dir / 'server.log'}; container retained.")
    (log_dir / "response.json").write_text(json.dumps(response, indent=2) + "\n")
    answers = response.get("answers") if isinstance(response, dict) else None
    answer = answers.get("q1") if isinstance(answers, dict) else None
    if (not isinstance(response, dict) or response.get("model") != pin["repo"]
            or not isinstance(answer, dict) or answer.get("type") != "choice"
            or answer.get("choice") not in ("A", "B", "C", "D")):
        save_logs(container, log_dir)
        fail(f"Unexpected decision response. See {log_dir / 'response.json'}; container retained.")
    response.pop("_benchmark", None)
    print(json.dumps(response, indent=2))
    print(f"API ready: {base}/v1/systemone\nLogs: {log_dir}")
    return 0


def save_logs(container: str, log_dir: Path) -> None:
    with (log_dir / "server.log").open("w") as output:
        subprocess.run(["docker", "logs", container], stdout=output, stderr=subprocess.STDOUT)


if __name__ == "__main__":
    try:
        sys.exit(launch())
    except (RuntimeError, subprocess.CalledProcessError, HTTPError, URLError, TimeoutError, OSError) as error:
        print(f"Launcher failed: {error}", file=sys.stderr)
        sys.exit(1)
