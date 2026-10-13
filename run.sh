#!/usr/bin/env bash
# Download (if needed), launch, and verify the pinned Clef Flash decision API.
set -euo pipefail
if [[ "${1:-}" == '--help' ]]; then
  echo 'Usage: ./run.sh [existing-pinned-clef-flash-directory]'
  echo 'Requires Linux ARM64, Docker GPU support, Python 3, curl and free GPU memory.'
  echo 'Optional: IMAGE, CONTAINER_NAME and PORT environment variables.'
  exit 0
fi
[[ $# -le 1 ]] || { echo "Usage: ./run.sh [model-directory]" >&2; exit 1; }
cd "$(dirname "$0")"
for dependency in docker python3 curl; do command -v "$dependency" >/dev/null; done
[[ "$(uname -s)" == Linux && "$(uname -m)" == aarch64 ]] || {
  echo 'This prebuilt image targets Linux ARM64 / DGX Spark.' >&2; exit 1;
}
IMAGE=${IMAGE:-ghcr.io/krisitown/local-jev-api:spark-20261013}
CONTAINER_NAME=${CONTAINER_NAME:-local-jev-flash}
PORT=${PORT:-18081}
[[ "$PORT" =~ ^[0-9]+$ ]] && (( PORT > 0 && PORT < 65536 )) || { echo 'Invalid PORT' >&2; exit 1; }
if docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
  echo "Container $CONTAINER_NAME already exists; inspect it or choose another CONTAINER_NAME." >&2
  exit 1
fi
# Pull an already-built runtime; this does not build or modify the host Python.
docker pull "$IMAGE"
if [[ $# -eq 0 ]]; then
  # The downloader reuses its cache while resolving the pinned snapshot revision.
  python3 ops/download_model.py --key clef-flash --image "$IMAGE"
  MODEL_DIR="$PWD/checkpoints/clef-flash"
else
  MODEL_DIR=$(cd "$1" && pwd)
fi
[[ -f "$MODEL_DIR/joint_schema_model.py" && -f "$MODEL_DIR/model.safetensors.index.json" ]] || {
  echo 'Expected the complete pinned Cloudflare/clef-flash snapshot.' >&2; exit 1;
}
# Never stop or reconfigure other workloads automatically.
docker run -d --name "$CONTAINER_NAME" --gpus all --ipc=host \
  -p "127.0.0.1:$PORT:18081" \
  -v "$MODEL_DIR:/model:ro" \
  -e MODEL_PATH=/model -e MODEL_ID=Cloudflare/clef-flash \
  -e MODEL_REVISION=8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3 \
  -e BACKEND=clef "$IMAGE"
mkdir -p logs
RUN_LOGS="logs/$(date +%Y%m%d-%H%M%S)-$$"
mkdir "$RUN_LOGS"
ready=0
for ((i=0; i<120; i++)); do
  if [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME")" != true ]]; then
    docker logs "$CONTAINER_NAME" > "$RUN_LOGS/server.log" 2>&1
    echo "Container exited. See $RUN_LOGS/server.log" >&2; exit 1
  fi
  if curl -fsS --max-time 3 "http://127.0.0.1:$PORT/health" -o "$RUN_LOGS/health.json" 2>/dev/null && \
    python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d.get("ready") and d.get("model")=="Cloudflare/clef-flash" else 1)' "$RUN_LOGS/health.json"; then
    ready=1; break
  fi
  sleep 5
 done
if [[ "$ready" != 1 ]]; then
  docker logs "$CONTAINER_NAME" > "$RUN_LOGS/server.log" 2>&1
  echo "Readiness timed out. Inspect $RUN_LOGS/server.log; container retained for diagnosis." >&2
  exit 1
fi
# Verify an actual typed response, not merely an open HTTP port.
curl -fsS --max-time 120 "http://127.0.0.1:$PORT/v1/systemone" \
  -H 'Content-Type: application/json' --data-binary @request.json \
  -o "$RUN_LOGS/response.json"
python3 - "$RUN_LOGS/response.json" <<'PY'
import json,sys
response=json.load(open(sys.argv[1]))
assert response.get('model')=='Cloudflare/clef-flash', 'Unexpected response model'
answer=response.get('answers',{}).get('q1',{})
assert answer.get('type')=='choice' and answer.get('choice') in ('A','B','C','D'), 'Expected a typed q1 choice'
response.pop('_benchmark',None)
print(json.dumps(response,indent=2))
PY
printf 'API ready: http://127.0.0.1:%s/v1/systemone\nLogs: %s\n' "$PORT" "$RUN_LOGS"
