#!/usr/bin/env bash
# Step 09 — Build llama-cpp-python on ppc64le with the Power10 MMA kernels, then download a GGUF model.
#
# Usage (project root, venv from Step 03 must exist):
#   ./scripts/09_setup_llamacpp.sh                                    # Llama-3.2-3B Q4_0 (default)
#   ./scripts/09_setup_llamacpp.sh Qwen/Qwen2.5-3B-Instruct-GGUF qwen2.5-3b-instruct-q8_0.gguf
set -euo pipefail
cd "$(dirname "$0")/.."

REPO=${1:-bartowski/Llama-3.2-3B-Instruct-GGUF}
FILE=${2:-Llama-3.2-3B-Instruct-Q4_0.gguf}
VERSION=${LLAMA_CPP_PYTHON_VERSION:-0.3.37}     # latest on PyPI in Nov 2026; pin what you tested
CPUTYPE=${GGML_CPUTYPE:-power10}                 # Power11 runs Power10 code; GCC 13 has no power11 target

# shellcheck disable=SC1091
source venv/bin/activate
mkdir -p models results

# 1. Use GCC 13 from gcc-toolset (installed in Step 02): mature -mcpu=power10 and MMA builtin support.
# shellcheck disable=SC1091
source /opt/rh/gcc-toolset-13/enable
echo "Compiler: $(gcc --version | head -1)"

# 2. Build from source. The CMake flags are passed through to llama.cpp's ggml library:
#    GGML_NATIVE=OFF + GGML_CPU_POWERPC_CPUTYPE=power10 -> compile with -mcpu=power10 explicitly
#      (deterministic: the build does not depend on parsing /proc/cpuinfo). This defines __MMA__,
#      which switches on ggml's MMA matrix kernels (tinyBLAS/llamafile sgemm) for Q4_0, Q8_0, bf16, fp32.
#    GGML_BLAS=OFF  -> use ggml's own kernels, not an external BLAS
#    LLAMA_CURL=OFF -> the Python package does not need libcurl
export CMAKE_ARGS="-DGGML_NATIVE=OFF -DGGML_CPU_POWERPC_CPUTYPE=${CPUTYPE} -DGGML_BLAS=OFF -DLLAMA_CURL=OFF"
export FORCE_CMAKE=1
echo "Building llama-cpp-python ${VERSION} with: ${CMAKE_ARGS}   (takes 5-15 minutes)"
pip install --no-cache-dir --no-binary llama-cpp-python --verbose "llama-cpp-python==${VERSION}" \
  > results/llamacpp-build.log 2>&1 || { tail -40 results/llamacpp-build.log; exit 1; }
echo "Build log: results/llamacpp-build.log"
grep -m1 -o -- "-mcpu=${CPUTYPE}" results/llamacpp-build.log >/dev/null \
  && echo "  [ OK ] compiled with -mcpu=${CPUTYPE}" || echo "  [WARN] -mcpu=${CPUTYPE} not found in build log"

# 3. Prove the MMA instructions are really inside the compiled CPU library.
#    xv*ger* are the Power10 MMA outer-product instructions (e.g. xvi8ger4pp, xvf32gerpp, xvbf16ger2pp).
LIB=$(python - <<'EOF'
import glob, os, llama_cpp
d = os.path.dirname(llama_cpp.__file__)
libs = glob.glob(os.path.join(d, "**", "libggml-cpu*.so*"), recursive=True) or \
       glob.glob(os.path.join(d, "**", "libggml*.so*"), recursive=True)
print(libs[0] if libs else "")
EOF
)
if [[ -n "$LIB" ]]; then
  n=$(objdump -d "$LIB" | grep -cE '\sxv(i8|i4|i16|f32|f64|bf16|f16)ger' || true)
  echo "  MMA instructions in $(basename "$LIB"): $n"
  (( n > 0 )) && echo "  [ OK ] MMA kernels present" || echo "  [WARN] no MMA instructions found"
fi
python -c "import llama_cpp; print('  llama_cpp', llama_cpp.__version__, '|', llama_cpp.llama_print_system_info().decode().strip())"

# 4. Download the GGUF file (the 'hf' CLI comes with huggingface_hub). Split files (*-00001-of-0000N.gguf)
#    need every part in the same folder; point gguf_path at the first part.
echo "Downloading ${REPO}/${FILE} -> models/"
hf download "$REPO" "$FILE" --local-dir models
ls -lh "models/${FILE}"

cat <<EOF

Done. To use it, edit config.yaml:
    backend: llamacpp
    llamacpp:
      gguf_path: models/${FILE}
Then verify:   python cli.py --selftest
EOF
