#!/usr/bin/env bash
# Start the Solution 08 API (used by common/deploy/ai-app@.service).
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source venv/bin/activate
exec python src/app.py
