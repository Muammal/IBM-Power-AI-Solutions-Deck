#!/usr/bin/env bash
# C4 — Build llama.cpp (llama-server, llama-cli, llama-quantize, llama-bench) for Power10/Power11
# with the MMA matrix kernels, and prove the MMA instructions are in the binary.
#
# Usage:  sudo -E common/scripts/build_llama_server.sh          # installs into /opt/llama.cpp
#         LLAMA_CPP_TAG=v1.0.0 LLAMA_CPP_DIR=$HOME/llama.cpp common/scripts/build_llama_server.sh
set -euo pipefail

TAG=${LLAMA_CPP_TAG:-v1.0.0}            # latest llama.cpp release on 2026-11-15. Pin what you tested.
DIR=${LLAMA_CPP_DIR:-/opt/llama.cpp}
CPUTYPE=${GGML_CPUTYPE:-power10}        # Power11 runs Power10 code; GCC 13 has no power11 target.

# GCC 13 from gcc-toolset (installed by os_prep.sh): mature -mcpu=power10 + MMA builtins.
# shellcheck disable=SC1091
source /opt/rh/gcc-toolset-13/enable
echo "Compiler: $(gcc --version | head -1)"

if [[ ! -d "$DIR/.git" ]]; then
  git clone --depth 1 --branch "$TAG" https://github.com/ggml-org/llama.cpp "$DIR"
fi
cd "$DIR"

# GGML_NATIVE=OFF + GGML_CPU_POWERPC_CPUTYPE=power10 -> compile with -mcpu=power10 explicitly.
#   That defines __MMA__, which turns on ggml's MMA kernels (llamafile/tinyBLAS sgemm)
#   for Q4_0, Q8_0, BF16 and F32 matrices. Prefer those GGUF types on Power.
# GGML_BLAS=OFF  -> ggml's own kernels (they already use MMA)
# LLAMA_CURL=OFF -> no libcurl needed; models are downloaded with `hf download`
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=OFF -DGGML_CPU_POWERPC_CPUTYPE="$CPUTYPE" -DGGML_BLAS=OFF -DLLAMA_CURL=OFF
cmake --build build -j"$(nproc)" --target llama-server llama-cli llama-quantize llama-bench

# Prove MMA is really compiled in: xv*ger* are the Power10 MMA outer-product instructions.
LIB=$(find build -name 'libggml-cpu*.so*' | head -1)
TARGET=${LIB:-build/bin/llama-server}
n=$(objdump -d "$TARGET" | grep -cE '\sxv(i8|i4|i16|f32|f64|bf16|f16)ger' || true)
echo "MMA instructions in $(basename "$TARGET"): $n"
(( n > 0 )) && echo "[ OK ] MMA kernels present" || { echo "[FAIL] no MMA instructions found"; exit 1; }

# Python deps for convert_hf_to_gguf.py (used after fine-tuning, see hf_to_gguf.sh)
echo
echo "Built: $DIR/build/bin/llama-server"
"$DIR/build/bin/llama-server" --version 2>&1 | head -2 || true
