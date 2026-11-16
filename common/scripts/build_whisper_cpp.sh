#!/usr/bin/env bash
# C6 — Build whisper.cpp (whisper-server, whisper-cli) for Power10/Power11 with MMA, download a model.
# Used by Solution 04 (speech-to-text).
#
# Usage:  sudo -E common/scripts/build_whisper_cpp.sh [ggml-model-file]
#         default model: ggml-large-v3-turbo-q8_0.bin (multilingual incl. Arabic, MIT licence)
set -euo pipefail
TAG=${WHISPER_CPP_TAG:-v1.9.5}          # latest whisper.cpp release on 2026-11-15
DIR=${WHISPER_CPP_DIR:-/opt/whisper.cpp}
MODEL=${1:-ggml-large-v3-turbo-q8_0.bin}
CPUTYPE=${GGML_CPUTYPE:-power10}

# shellcheck disable=SC1091
source /opt/rh/gcc-toolset-13/enable
[[ -d "$DIR/.git" ]] || git clone --depth 1 --branch "$TAG" https://github.com/ggml-org/whisper.cpp "$DIR"
cd "$DIR"
# whisper.cpp vendors the same ggml library as llama.cpp, so the same Power flags apply.
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=OFF -DGGML_CPU_POWERPC_CPUTYPE="$CPUTYPE" -DGGML_BLAS=OFF -DWHISPER_BUILD_SERVER=ON
cmake --build build -j"$(nproc)" --target whisper-server whisper-cli

LIB=$(find build -name 'libggml-cpu*.so*' | head -1)
n=$(objdump -d "${LIB:-build/bin/whisper-cli}" | grep -cE '\sxv(i8|f32|bf16|f16)ger' || true)
echo "MMA instructions: $n"; (( n > 0 )) && echo "[ OK ] MMA kernels present" || echo "[WARN] none found"

mkdir -p /opt/models
hf download ggerganov/whisper.cpp "$MODEL" --local-dir /opt/models
ls -lh "/opt/models/$MODEL"
