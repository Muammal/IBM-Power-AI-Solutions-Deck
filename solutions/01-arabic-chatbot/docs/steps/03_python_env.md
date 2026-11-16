# Step 03 — Python environment from IBM wheels

## Goal

Create a Python 3.12 virtual environment and install **pinned** packages, taking PyTorch and
other native packages from **IBM's Power-optimized wheel index**. Those wheels are built for
ppc64le with Power10 MMA support. A generic `pip install torch` has no ppc64le build on PyPI.

The install pattern (`python3.12 -m venv`, `--prefer-binary`, dated IBM index) and the pins
come from the official examples in
[ppc64le/pyeco `examples/Python3.12`](https://github.com/ppc64le/pyeco/tree/v2026.03.31/examples/Python3.12)
(mainly `Granite-with-pytorch-example` and `vllm-example`).

## Files created in this step

| File | Purpose |
|------|---------|
| `requirements.txt` | IBM extra index at the top, every direct dependency pinned |
| `scripts/03_setup_env.sh` | Creates `venv/`, installs `requirements.txt`, prints versions |
| `docs/steps/03_python_env.md` | This document |

## The IBM index, explained

```text
--extra-index-url https://wheels.developerfirst.ibm.com/ppc64le/linux-v2026.03.31/+simple/
--prefer-binary
```

* `--extra-index-url`: pip looks at PyPI **and** the IBM index, and picks the best match.
  Pins such as `torch==2.10.0+ppc64le1` exist only on the IBM index.
* `linux-v2026.03.31`: a **dated snapshot**. It does not change, so installs are reproducible.
  The moving index `.../ppc64le/linux/+simple/` has newer builds (for example torch 2.14).
  Use it only when you want to upgrade on purpose.
* `/+simple/`: the server runs *devpi*. With this suffix pip gets the package list. Without it
  pip can receive the human web page and report `from versions: none`.
* `--prefer-binary`: never pick a source package when a wheel exists.

Ad-hoc installs use the same flags:

```bash
pip install --prefer-binary <pkgs> --extra-index-url=https://wheels.developerfirst.ibm.com/ppc64le/linux-v2026.03.31/+simple/
```

## Commands to run (copy-paste ready)

Run as your normal user (not root), from the project root:

```bash
cd /opt/power-chatbot
./scripts/03_setup_env.sh
source venv/bin/activate
```

## Expected output

The last lines of the script:

```
== Verification
2.10.0+ppc64le1
transformers 4.57.6 | fastapi 0.135.2
torch threads: 32 | bf16 supported: True        <- e.g. 8 cores x SMT4: PyTorch's own default on POWER

Done. Activate the environment in new shells with:  source venv/bin/activate
```

`torch threads` here is PyTorch's own default. **On POWER it is one thread per SMT hardware thread** (32 on 8 cores
in SMT4), because PyTorch's core-detection library (`cpuinfo`) does not support POWER. The chatbot does not use this
default: with `runtime.threads: 0`, `llm.py` counts the physical cores and uses one thread per core (Step 05). You tune it in Step 08.

### What the IBM torch wheel contains (inspected while building this template)

| Item | Finding | Meaning |
|---|---|---|
| `libtorch_cpu.so` build flags | `-mcpu=power9 -mtune=power10 -O2` | PyTorch's own kernels (element-wise ops, softmax, ...) use Power9 VSX vector instructions and run on Power9, 10 and 11 |
| Bundled BLAS | `torch.libs/libopenblasp-r0-*.3.29.so` (OpenBLAS 0.3.29, OpenMP build) with `POWER8/9/10` kernel sets | OpenBLAS picks the **POWER10 kernels at run time** on Power10/11. These are the MMA-based kernels |
| BLAS calls from PyTorch | imports `sgemm_` (float32) and `sbgemm_` (bfloat16) | The big matrix multiplies of an LLM (the linear layers) go to OpenBLAS, so both float32 and bfloat16 can use MMA |
| Threading | OpenBLAS uses the same OpenMP runtime (`libgomp`) as PyTorch | `runtime.threads` / `OMP_NUM_THREADS` controls the matrix multiplies too |

What was **not** proven here: that every decode-time multiply takes the `sbgemm_` path (single-token decoding is
matrix × vector, and PyTorch may use its own code for some bf16 shapes). The bf16-vs-fp32 benchmark in Step 08 is how you find out on your LPAR.

## How to verify

```bash
source venv/bin/activate
python -c "import torch, transformers; print(torch.__version__)"
```

**Passes when it prints `2.10.0+ppc64le1`.** A version without `+ppc64le` means torch came
from somewhere else. Remove the venv and run the script again.

Optional: confirm that every installed package is consistent:

```bash
pip check
```

> Checked while building this template: `requirements.txt` resolves to **40 packages, all binary
> wheels**, for ppc64le / CPython 3.12 / glibc 2.34 (RHEL 9). The IBM index supplies `torch`,
> `numpy`, `sentencepiece`, `PyYAML`, `hf-xet` and `mpmath`; everything else is a standard PyPI wheel.

## Common errors and fixes

| Symptom | Cause | Fix |
|---------|-------|-----|
| `No matching distribution found for torch==2.10.0+ppc64le1 (from versions: none)` | Index URL missing `/+simple/`, or no internet/proxy | Use the exact first line of `requirements.txt`; set `HTTPS_PROXY` if needed |
| `... manylinux_2_34_ppc64le.whl is not a supported wheel` | OS older than RHEL 9 (glibc < 2.34) | Use RHEL 9 |
| `python3.12: command not found` | Step 02 not done | `sudo ./scripts/02_os_prep.sh` |
| pip starts compiling (`Building wheel for ...`) or fails with `--only-binary` | A package has no ppc64le wheel at that version | Look up the available version: `pip index versions <pkg> --extra-index-url <IBM index>` and adjust the pin |
| `ImportError: libopenblas.so...` / `libgomp` | Missing system library | `sudo dnf install -y openblas libgomp` |
| Disk fills up during install | pip cache | The script already uses `--no-cache`; clean old caches with `pip cache purge` |
