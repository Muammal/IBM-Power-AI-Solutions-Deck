#!/usr/bin/env bash
# Start the Solution 11 service (used by common/deploy/ai-app@.service).
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -f tokens.yaml ]] || { echo "tokens.yaml missing: see tokens.yaml.example"; exit 1; }
[[ -f data/inventory.csv ]] || echo "WARNING: data/inventory.csv missing: CISA KEV matching is off (see data/inventory.example.csv)"
# shellcheck disable=SC1091
source venv/bin/activate
exec python src/app.py
