#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODELS_DIR="$ROOT/models"
REPO_ID="2vhoc/helmet-detection-traffic"

echo "Downloading models from Hugging Face: $REPO_ID"
echo "Destination: $MODELS_DIR"

python3 - <<PY
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="${REPO_ID}",
    local_dir="${MODELS_DIR}",
    local_dir_use_symlinks=False,
)
print("Done.")
PY

echo "Models ready under $MODELS_DIR"
