#!/usr/bin/env bash
# Delete stored call results older than storage.retention_days (run daily from cron or a systemd timer).
# Usage: scripts/purge_old_calls.sh [days]
set -euo pipefail
cd "$(dirname "$0")/.."
DAYS=${1:-$(awk '/retention_days:/ {print $2}' config.yaml)}
find data/calls -name '*.json' -type f -mtime +"$DAYS" -print -delete
