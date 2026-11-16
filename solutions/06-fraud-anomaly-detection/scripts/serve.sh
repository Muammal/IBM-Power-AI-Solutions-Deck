#!/usr/bin/env bash
# Start the Solution 06 scoring API (used by common/deploy/ai-app@.service).
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source venv/bin/activate
exec python src/score_api.py
