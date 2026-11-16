<div dir="rtl">

## ملخص بالعربية

هذا المشروع **قالب تعليمي بسيط** لروبوت محادثة (Chatbot) يعمل على **المعالج المركزي فقط (CPU)** في خوادم
**IBM Power10 / Power11** بنظام **RHEL 9** و **Python 3.12**، باستخدام مكتبة **Hugging Face transformers**.

- يفهم **العربية الفصحى واللهجة العراقية** والإنجليزية، ويرد تلقائياً بلغة المستخدم.
- **تغيير النموذج لا يحتاج أي تعديل في الكود**: كل الإعدادات في ملف واحد هو `config.yaml`
  (Llama 3.2 أو Qwen2.5 أو ALLaM).
- يستخدم **قالب المحادثة الخاص بكل نموذج** (`apply_chat_template`) ولا يكتب صيغة الأوامر يدوياً.
- **بث الإجابة كلمة بكلمة** في سطر الأوامر وفي واجهة ويب تدعم الكتابة من اليمين إلى اليسار.
- محرك اختياري ثانٍ: **llama.cpp** مع ملفات GGUF (Q4_0 / Q8_0) المسرّعة بوحدات **MMA** في Power10.
- الحزم تُثبَّت من **مستودع IBM المحسّن لمعالجات Power**، بنفس طريقة أمثلة
  [ppc64le/pyeco](https://github.com/ppc64le/pyeco/tree/v2026.03.31/examples/Python3.12).

اتبع الخطوات من 01 إلى 11 بالترتيب. كل خطوة تنتهي بأمر تحقق، فلا تنتقل للخطوة التالية قبل نجاحه.

</div>

---

# Solution 01 — Power Chatbot: a CPU-only Arabic/English LLM template for IBM Power10/Power11

> **Update 2026-11-15 (part of the IBM Power AI Solutions repo):** the default model is now
> **`ibm-granite/granite-4.2-3b`** (Apache-2.0, not gated, Arabic tested), with Granite's thinking mode turned off
> through `model.chat_template_kwargs` in `config.yaml`. The Llama/Qwen/ALLaM presets are still there.
> See also [`IMPLEMENTATION.md`](IMPLEMENTATION.md) (IBM Cloud vs on-prem path and fine-tuning),
> [`RESOURCES.md`](RESOURCES.md), [`AGENTS.md`](AGENTS.md) and the Word versions in `docs/`.

A small, well-commented chatbot that colleagues can **learn from and reuse**: Hugging Face `transformers`
on CPU (Linux ppc64le, RHEL 9, Python 3.12), with an optional llama.cpp backend. The model is chosen in
`config.yaml` only.

## Architecture

```
                         +----------------------- config.yaml ------------------------+
                         | backend, model.id, dtype, threads, generation, system_prompt |
                         +------------------------------+-----------------------------+
                                                        | read by
   +-------------+    POST /chat (JSON)    +------------v------------+
   |  Browser    | ----------------------> |  app.py  (FastAPI)      |    +--------------------------+
   | index.html  | <---------------------- |  /  /health  /chat      |--->|                          |
   | dir="auto"  |   streamed UTF-8 text   |  one answer at a time   |    |  llm.py                  |
   +-------------+                         +-------------------------+    |  load_config()           |
                                                                          |  load_model(cfg)         |
   +-------------+   same function calls   +-------------------------+    |  generate(messages,      |
   |  Terminal   | <---------------------> |  cli.py  (--selftest)   |--->|           stream=True)   |
   +-------------+                         +-------------------------+    |                          |
                                           +-------------------------+    |  + system prompt (AR/EN) |
                                           | scripts/benchmark.py    |--->|  + history trimming      |
                                           +-------------------------+    +------+------------+------+
                                                                                 |            |
                                    backend: transformers                       |            |  backend: llamacpp
                    +-----------------------------------------------------------+            +-------------------------------+
                    | AutoTokenizer.apply_chat_template  (model's own template) |            | llama_cpp.Llama(gguf)         |
                    | AutoModelForCausalLM  (bf16 / fp32, torch threads)        |            | create_chat_completion(stream)|
                    | TextIteratorStreamer + generate() thread                  |            | (template stored in the GGUF) |
                    +-----------------------------+-----------------------------+            +---------------+---------------+
                                                  |                                                          |
     PyTorch 2.10 (IBM wheel) -> matmuls in bundled OpenBLAS 0.3.29 POWER10 kernels      ggml with -mcpu=power10 (MMA Q4_0/Q8_0)
     threads = one per physical core (llm.physical_cores)
                                                  +--------------------------+-------------------------------+
                                                                             |
                                                  IBM Power10 / Power11 LPAR, RHEL 9, SMT4
```

## Step-by-step guide (do them in order)

| Step | Document | What you get | Verification |
|---:|---|---|---|
| 01 | [Provision the Power LPAR](docs/steps/01_provision_power.md) | RHEL 9 LPAR on PowerVS (S1122) or on-premises, sized for 3B/7B/8B | `uname -m` = `ppc64le` |
| 02 | [Prepare the OS and check MMA](docs/steps/02_verify_mma.md) | Python 3.12, gcc-toolset-13, cmake, git, SMT4 | `./scripts/check_power.sh` → `RESULT: PASS` |
| 03 | [Python environment from IBM wheels](docs/steps/03_python_env.md) | `venv/` with pinned packages from IBM's ppc64le index | prints `2.10.0+ppc64le1` |
| 04 | [Configuration and choosing a model](docs/steps/04_choose_model.md) | `config.yaml`, `.env`, preset table (size, RAM, Arabic, licence) | model info line printed |
| 05 | [Model layer](docs/steps/05_llm_module.md) | `llm.py` explained line by line | EN + AR answers |
| 06 | [Command-line smoke test](docs/steps/06_cli_test.md) | `cli.py` chat + `--selftest` | `SELFTEST PASSED` |
| 07 | [Web UI](docs/steps/07_web_ui.md) | `app.py` + `static/index.html`, RTL + streaming + Stop | browser checks |
| 08 | [Benchmark and tuning](docs/steps/08_benchmark_tuning.md) | load time, TTFT, tok/s for dtype × threads → CSV | CSV rows written |
| 09 | [Optional llama.cpp backend](docs/steps/09_llamacpp_backend.md) | `llama-cpp-python` built with Power10 MMA, GGUF Q4_0/Q8_0 | same `cli.py --selftest` passes |
| 10 | [Run as a service](docs/steps/10_deploy.md) | systemd unit or Podman container | `/health` after restart |
| 11 | [**How to switch models**](docs/steps/11_swap_models.md) | Llama → Qwen → ALLaM → GGUF by editing `config.yaml` only | selftest with new model |

## Quick start (on the LPAR, after Steps 01–02)

```bash
cd /opt/power-chatbot
./scripts/03_setup_env.sh && source venv/bin/activate
cp -n .env.example .env && vi .env        # HF_TOKEN=... (needed for the default Llama model)
python cli.py --selftest                  # Arabic + English smoke test
python app.py                             # then: ssh -N -L 7860:127.0.0.1:7860 user@lpar → http://localhost:7860
```

## Project layout

```
power-chatbot/
├── app.py                 FastAPI web server (one page, streaming)          ~100 lines
├── cli.py                 terminal chat + --selftest                         ~105 lines
├── llm.py                 model layer: both backends, same interface         ~185 lines
├── config.yaml            THE file you edit: model, backend, dtype, threads, prompts
├── .env.example           HF_TOKEN and optional overrides (copy to .env)
├── requirements.txt       pinned; IBM ppc64le wheel index on the first line
├── static/index.html      chat page, dir="auto", no framework
├── scripts/
│   ├── 02_os_prep.sh      dnf packages, optional SMT4
│   ├── check_power.sh     arch / MMA / SMT / RAM / Python check (exit code)
│   ├── 03_setup_env.sh    venv + pip install from IBM wheels
│   ├── benchmark.py       load time, TTFT, tokens/s → results/benchmark.csv
│   └── 09_setup_llamacpp.sh  build llama-cpp-python for Power10 + download GGUF
├── deploy/
│   ├── Containerfile      UBI 9 Python 3.12 (ppc64le), Podman
│   └── chatbot.service    systemd unit (hardened, offline Hub)
└── docs/steps/01 … 11     one document per step
```

## Things we found while building this (and what the template does about them)

| Finding | Consequence |
|---|---|
| `https://wheels.developerfirst.ibm.com/ppc64le/linux` (and the dated `linux-v2026.03.31` used in pyeco) is a **devpi** server. Requested without `/+simple/`, pip received the human web page and found **no** packages | `requirements.txt` uses `.../linux-v2026.03.31/+simple/` |
| The IBM snapshot has **no Gradio** wheel; FastAPI + uvicorn are already pinned in pyeco's `vllm-example` | Web UI is FastAPI + one HTML page |
| ALLaM's repo moved to **`humain-ai/ALLaM-7B-Instruct-preview`** | Preset uses the new ID |
| **Qwen2.5-3B** is under the *Qwen Research Licence* (non-commercial); Qwen2.5-7B and ALLaM are Apache-2.0; Llama 3.2 is gated | Spelled out in `config.yaml` and Step 04 |
| transformers 4.57 contacts the Hub on **every** tokenizer load, even from cache. A network blip crashed start-up | Service runs with `HF_HUB_OFFLINE=1`; documented in Steps 07/10 |
| A first version kept generating ~15 s after a browser disconnected | `generate(..., stop=Event)` + disconnect check: lock released in < 1 s |
| Inside IBM's torch wheel: `libtorch_cpu` is built `-mcpu=power9 -mtune=power10`, and calls `sgemm_`/`sbgemm_` in a bundled **OpenBLAS 0.3.29** (OpenMP build) that contains POWER10 kernels | The heavy fp32/bf16 matrix multiplies can use MMA; Step 08's bf16-vs-fp32 benchmark confirms it on your LPAR |
| PyTorch cannot count cores on POWER (its `cpuinfo` helper has no POWER support), so it defaults to **one thread per SMT thread** | `runtime.threads: 0` now means one thread per physical core (from `/sys`), and `OMP_NUM_THREADS` is set before torch loads |
| Big LPARs can span several NUMA nodes | `check_power.sh` reports it; Steps 08/10 show `numactl` pinning |

## What was verified, and where

The target hardware (Power10/11) was **not available** while building this template. The verification was:

| Item | How it was verified | Result |
|---|---|---|
| `requirements.txt` | Full pip resolution for **ppc64le / CPython 3.12 / glibc 2.34**, binary-only, against IBM index + PyPI | 40 packages, all wheels; torch/numpy/sentencepiece/PyYAML from IBM |
| `llm.py`, `cli.py` | Run on x86 in a venv with the **same pinned versions** (torch 2.10.0, transformers 4.57.6), stand-in models Qwen2.5-0.5B/1.5B-Instruct | EN + AR + streaming OK; 1.5B: `SELFTEST PASSED` (MSA, Iraqi, English); 0.5B fails Iraqi (correctly detected) |
| `app.py` + `index.html` | Browser test | RTL for Arabic bubbles, LTR for English, 39 streamed updates, Stop frees the model, 422/400 validation |
| `scripts/benchmark.py` | 0.5B model, 2 dtypes × 3 thread counts | 12 CSV rows |
| llama.cpp backend | x86 prebuilt llama-cpp-python 0.3.36 + `qwen2.5-0.5b-instruct-q4_0.gguf`, unchanged `cli.py`, config differing only in `backend:` | Loads, streams, GGUF template used, stop event works, MSA + English pass (Iraqi fails with 0.5B, same as transformers); ~90 tok/s |
| Power build of PyTorch | IBM wheel `torch-2.10.0+ppc64le1` downloaded (sha256 matches the index) and inspected: build flags, bundled libraries, imported BLAS symbols | See findings above. Not executed: needs ppc64le |
| Core counting (`physical_cores`) | Simulated 8-core SMT4 sysfs layout; affinity limited to 6 threads | 8 cores; 2 cores. Falls back to the library default off Linux |
| Shell scripts | `bash -n` syntax check | OK. Not executed: they need RHEL 9 on ppc64le |
| `Containerfile`, `chatbot.service`, PowerVS commands, MMA build flags | Written from documentation; **not executed** | Run Steps 01, 02, 09 and 10 on the LPAR to verify |

## Credits

Install pattern, IBM wheel index snapshot and package pins follow the official
[ppc64le/pyeco Python 3.12 examples (v2026.03.31)](https://github.com/ppc64le/pyeco/tree/v2026.03.31/examples/Python3.12),
especially `Granite-with-pytorch-example` and `vllm-example` (Apache-2.0).
