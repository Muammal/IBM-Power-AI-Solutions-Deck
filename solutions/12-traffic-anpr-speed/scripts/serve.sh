#!/usr/bin/env bash
# Start the Solution 12 service (used by common/deploy/ai-app@.service).
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -f tokens.yaml ]] || { echo "tokens.yaml missing: see tokens.yaml.example"; exit 1; }
for m in models/detector/model.onnx models/ocr/plate_ocr.onnx; do
  [[ -f "$m" ]] || { echo "$m missing: IMPLEMENTATION.md Steps 5-6"; exit 1; }
done
grep -q '^PLATE_HMAC_KEY=' .env 2>/dev/null || { echo "set PLATE_HMAC_KEY in .env (openssl rand -hex 32)"; exit 1; }
# shellcheck disable=SC1091
source venv/bin/activate
exec python src/app.py
