# Step 09 — Optional llama.cpp backend

## Goal

Add a second inference engine, **llama.cpp** through `llama-cpp-python`, that runs quantized **GGUF** files.
Then switch to it with **one line in `config.yaml`**, while `cli.py`, `app.py` and the benchmark keep working unchanged.

Why bother?

* **Much less RAM.** A 3B model is ~1.9 GB as Q4_0 and ~3.4 GB as Q8_0, versus ~6.4 GB in bf16.
* **Fast on Power.** llama.cpp has kernels that use the Power10/Power11 **MMA** units for Q4_0 and Q8_0
  (as well as fp32/bf16). That is why this template uses **Q4_0 / Q8_0**, not the K-quants (Q4_K_M, ...), which use slower code paths.
* Q8_0 is close to bf16 quality. Q4_0 is smaller and faster, with a small quality loss that is often more visible in Arabic,
  so test both.

## Files created in this step

| File | Purpose |
|------|---------|
| `scripts/09_setup_llamacpp.sh` | Builds `llama-cpp-python` from source with `-mcpu=power10`, checks for MMA instructions, downloads a GGUF |
| `docs/steps/09_llamacpp_backend.md` | This document |

`llm.py` already contains the llama.cpp code (`_stream_llamacpp`), written in Step 05.

## How the backend keeps the same interface

| | `transformers` | `llamacpp` |
|---|---|---|
| Model file | safetensors from the Hub (`model.id`) | one `.gguf` file (`llamacpp.gguf_path`) |
| Chat template | `tokenizer.apply_chat_template` (from `tokenizer_config.json`) | `create_chat_completion` uses the template **stored inside the GGUF** (`tokenizer.chat_template` metadata) |
| Streaming | `TextIteratorStreamer` + thread | `stream=True` returns chunks |
| Output of `llm.generate()` | iterator of text pieces | iterator of text pieces |

Neither path hard-codes a prompt format.

## Commands to run (copy-paste ready)

```bash
cd /opt/power-chatbot && source venv/bin/activate
./scripts/09_setup_llamacpp.sh
```

Other GGUF files (confirm the file names on the repo's "Files" tab):

```bash
# Qwen2.5-3B, Q8_0 (higher quality)
./scripts/09_setup_llamacpp.sh Qwen/Qwen2.5-3B-Instruct-GGUF qwen2.5-3b-instruct-q8_0.gguf
# Qwen2.5-7B, Q4_0: split into 2 parts, download both, point gguf_path at part 00001
hf download Qwen/Qwen2.5-7B-Instruct-GGUF --include "qwen2.5-7b-instruct-q4_0-*.gguf" --local-dir models
```

> File names checked on Hugging Face while building this template: `bartowski/Llama-3.2-3B-Instruct-GGUF` has
> `Llama-3.2-3B-Instruct-Q4_0.gguf` and `...-Q8_0.gguf`; `Qwen/Qwen2.5-3B-Instruct-GGUF` has
> `qwen2.5-3b-instruct-q4_0.gguf` and `...-q8_0.gguf`; the Qwen 7B files are split into parts. For ALLaM there is no
> official GGUF, only community conversions (for example `Omartificial-Intelligence-Space/ALLaM-7B-Instruct-preview-Q8_0-GGUF`).
> Check who made a community file before you use it.

Then switch the backend. This is the only change:

```yaml
# config.yaml
backend: llamacpp
llamacpp:
  gguf_path: models/Llama-3.2-3B-Instruct-Q4_0.gguf
```

## Expected output

```
Compiler: gcc (GCC) 13.x ...
Building llama-cpp-python 0.3.37 with: -DGGML_NATIVE=OFF -DGGML_CPU_POWERPC_CPUTYPE=power10 ...   (takes 5-15 minutes)
Build log: results/llamacpp-build.log
  [ OK ] compiled with -mcpu=power10
  MMA instructions in libggml-cpu.so: 1234
  [ OK ] MMA kernels present
  llama_cpp 0.3.37 | CPU : VSX = 1 | LLAMAFILE = 1 | OPENMP = 1 | ...
Downloading bartowski/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-Q4_0.gguf -> models/
-rw-r--r--. 1 user user 1.9G ... models/Llama-3.2-3B-Instruct-Q4_0.gguf
```

The MMA count shows how many Power10 matrix (`xv…ger…`) instructions the compiler emitted. The exact number varies by
version. It just has to be greater than 0.

## How to verify

After switching `backend: llamacpp` in `config.yaml`, run **the same `cli.py` as in Step 06, unchanged**:

```bash
python cli.py --selftest; echo "exit code: $?"
```

**Passes when** the header line starts with `Ready in ...: llamacpp | <your file>.gguf` and the result is
`SELFTEST PASSED` / `exit code: 0`. Then benchmark it to compare with transformers:

```bash
python scripts/benchmark.py --threads 8 16 32
```

> Tested while building this template on x86 (a prebuilt llama-cpp-python 0.3.36 CPU wheel, since the Power build
> needs a ppc64le machine) with `qwen2.5-0.5b-instruct-q4_0.gguf`. Only the `backend:` line differed from the
> transformers config. The unchanged `cli.py --selftest` loaded the GGUF in 0.5 s, used the file's own chat template,
> and passed MSA (`عاصمة العراق هي بغداد.`) and English. The Iraqi-dialect question failed with this 0.5B model, exactly
> as on the transformers backend, so the cause is the tiny model, not the backend. The `stop` event ended an answer after
> 5 pieces. The benchmark measured ~85–90 tok/s decode (vs ~12 tok/s for the same model in transformers bf16 on that laptop).
>
> Note for benchmarking: llama.cpp reuses the KV cache when a prompt starts with the same tokens as the previous
> one. Repeated identical benchmark prompts therefore show a near-zero TTFT. Treat the TTFT of the **first** question
> in a real chat as the honest number.

## Common errors and fixes

| Symptom | Cause | Fix |
|---------|-------|-----|
| `CMake Error: ... CMAKE_CXX_COMPILER not set` | No C++ compiler | `sudo dnf install -y gcc-c++ gcc-toolset-13` (Step 02) |
| `/opt/rh/gcc-toolset-13/enable: No such file` | gcc-toolset not installed | `sudo dnf install -y gcc-toolset-13` |
| `unrecognized command-line option '-mcpu=power10'` | Very old GCC used | Make sure the script's `source /opt/rh/gcc-toolset-13/enable` ran |
| `[WARN] no MMA instructions found` | Built for an older CPU type, or a cached wheel was reused | `pip uninstall -y llama-cpp-python` then rerun the script (it uses `--no-cache-dir --no-binary`) |
| `Illegal instruction` at runtime | Library built for power10 but running on Power9 or in P9 compat mode | `GGML_CPUTYPE=power9 ./scripts/09_setup_llamacpp.sh` (no MMA), or fix the LPAR mode |
| `ValueError: Failed to load model from file` | Wrong `gguf_path`, missing split parts, or a GGUF newer than the library | Check the path; download all parts; upgrade `LLAMA_CPP_PYTHON_VERSION` |
| Answers in a strange format / never stop | GGUF without a chat template in its metadata | Use GGUF files from the model author or a well-known converter |
| `ModuleNotFoundError: llama_cpp` in `app.py` | Built in a different venv | `source venv/bin/activate` before running the script |
