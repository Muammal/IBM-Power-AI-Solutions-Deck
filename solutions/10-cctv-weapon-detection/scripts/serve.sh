#!/usr/bin/env bash
# Start the Solution 10 service (used by common/deploy/ai-app@.service).
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -f tokens.yaml ]] || { echo "tokens.yaml missing: see tokens.yaml.example"; exit 1; }
[[ -f models/detector/model.onnx ]] || { echo "models/detector/model.onnx missing: IMPLEMENTATION.md Steps 5-7"; exit 1; }
# shellcheck disable=SC1091
source venv/bin/activate
exec python src/app.py
