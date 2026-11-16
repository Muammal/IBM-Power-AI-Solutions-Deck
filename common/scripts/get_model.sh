#!/usr/bin/env bash
# Download one file (GGUF / ggml) or a whole Hugging Face repo into /opt/models.
# Usage:  common/scripts/get_model.sh ibm-granite/granite-4.2-8b-GGUF granite-4.2-8b-Q4_0.gguf
#         common/scripts/get_model.sh ibm-granite/granite-embedding-278m-multilingual      # whole repo
# Needs the `hf` CLI (comes with huggingface_hub, installed in every solution venv).
set -euo pipefail
REPO=${1:?repo id}
FILE=${2:-}
DEST=${MODELS_DIR:-/opt/models}
mkdir -p "$DEST"
if [[ -n "$FILE" ]]; then
  hf download "$REPO" "$FILE" --local-dir "$DEST"
  ls -lh "$DEST/$FILE"
else
  hf download "$REPO" --local-dir "$DEST/${REPO#*/}"
  du -sh "$DEST/${REPO#*/}"
fi
