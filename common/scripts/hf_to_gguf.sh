#!/usr/bin/env bash
# C8 — Turn a fine-tuned (merged) Hugging Face model into a quantized GGUF for llama-server.
#
# Usage:  common/scripts/hf_to_gguf.sh out/merged-model /opt/models/my-model [Q4_0|Q8_0]
# Result: /opt/models/my-model-Q4_0.gguf
# Q4_0 and Q8_0 are the GGUF types with Power10 MMA kernels in ggml: prefer them on Power.
set -euo pipefail
SRC=${1:?merged HF model folder}
OUT=${2:?output path prefix (no extension)}
QUANT=${3:-Q4_0}
DIR=${LLAMA_CPP_DIR:-/opt/llama.cpp}

# The converter needs the gguf package shipped inside the llama.cpp tree (pure Python).
pip install --quiet "$DIR/gguf-py"
python "$DIR/convert_hf_to_gguf.py" "$SRC" --outtype bf16 --outfile "${OUT}-bf16.gguf"
"$DIR/build/bin/llama-quantize" "${OUT}-bf16.gguf" "${OUT}-${QUANT}.gguf" "$QUANT"
ls -lh "${OUT}"-*.gguf
echo "Serve it with: common/scripts/run_llama_server.sh ${OUT}-${QUANT}.gguf"
