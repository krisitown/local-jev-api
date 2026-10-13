#!/usr/bin/env bash
# Run a pinned decision checkpoint on Linux ARM64 / DGX Spark.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
exec python3 "$SCRIPT_DIR/ops/launch.py" "$@"
