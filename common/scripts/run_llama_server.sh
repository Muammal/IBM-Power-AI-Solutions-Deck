#!/usr/bin/env bash
# C5 — Start llama-server (OpenAI-compatible API) on CPU, tuned for Power.
#
# Usage:  common/scripts/run_llama_server.sh /opt/models/granite-4.2-8b-Q4_0.gguf [port] [ctx]
# Env:    THREADS (default: physical cores)   NUMA_NODE (pin to one node)   PARALLEL (slots, default 2)
#         LLAMA_CPP_DIR (default /opt/llama.cpp)   HOST (default 127.0.0.1)
set -euo pipefail
GGUF=${1:?path to .gguf}
PORT=${2:-8080}
CTX=${3:-16384}
DIR=${LLAMA_CPP_DIR:-/opt/llama.cpp}
HOST=${HOST:-127.0.0.1}
PARALLEL=${PARALLEL:-2}

# One thread per PHYSICAL core. With SMT4/SMT8 the kernel shows 4-8 "CPUs" per core,
# but the MMA units are per core: more threads than cores usually slows token generation.
cores=$(( $(lscpu | awk -F: '/^Core\(s\) per socket/ {print $2+0}') * $(lscpu | awk -F: '/^Socket\(s\)/ {print $2+0}') ))
THREADS=${THREADS:-$cores}

PIN=()
if [[ -n "${NUMA_NODE:-}" ]]; then PIN=(numactl --cpunodebind="$NUMA_NODE" --membind="$NUMA_NODE"); fi

# --jinja          : use the chat template stored in the GGUF (needed for tool calling and
#                    for Granite 4.2's enable_thinking switch, sent per request in chat_template_kwargs)
# --ctx-size       : total context shared by all slots (PARALLEL)
# --threads-batch  : prompt processing can use SMT threads profitably; generation cannot
exec "${PIN[@]}" "$DIR/build/bin/llama-server" \
  --model "$GGUF" --host "$HOST" --port "$PORT" \
  --ctx-size "$CTX" --parallel "$PARALLEL" \
  --threads "$THREADS" --threads-batch "$(nproc)" \
  --jinja --metrics
