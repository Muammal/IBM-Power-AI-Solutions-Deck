#!/usr/bin/env bash
# Start the Solution 04 API (used by common/deploy/ai-app@.service). whisper-server runs as its own service.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source venv/bin/activate
exec python src/app.py
