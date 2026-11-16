#!/usr/bin/env bash
# Start whisper-server (whisper.cpp, built in common step C6) on CPU with Power10 MMA kernels.
# Usage: scripts/run_whisper_server.sh [model.bin] [port]
set -euo pipefail
MODEL=${1:-/opt/models/ggml-large-v3-turbo-q8_0.bin}
PORT=${2:-8090}
DIR=${WHISPER_CPP_DIR:-/opt/whisper.cpp}
cores=$(( $(lscpu | awk -F: '/^Core\(s\) per socket/ {print $2+0}') * $(lscpu | awk -F: '/^Socket\(s\)/ {print $2+0}') ))
# -t: threads = physical cores; -l ar: Arabic (Iraqi dialect is still "ar");
# -bs 5: beam search, better for dialect at some CPU cost (use -bs 1 for speed)
exec "$DIR/build/bin/whisper-server" -m "$MODEL" --host 127.0.0.1 --port "$PORT" \
  -t "${THREADS:-$cores}" -l "${ASR_LANG:-ar}" -bs "${BEAM:-5}"
