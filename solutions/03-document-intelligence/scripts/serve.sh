#!/usr/bin/env bash
# Start the Solution 03 API (used by common/deploy/ai-app@.service).
set -euo pipefail
cd "$(dirname "$0")/.."
# Prefer the Docling venv when it exists (better tables), else the core venv.
VENV=venv; [[ -d venv-docling ]] && VENV=venv-docling
# shellcheck disable=SC1091
source "$VENV/bin/activate"
exec python src/app.py
