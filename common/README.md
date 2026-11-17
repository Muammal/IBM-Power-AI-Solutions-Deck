# Common platform guide (steps C1–C8)

Every solution in this repo starts with the same platform steps. Do them **once per LPAR**, then follow
the solution's own `IMPLEMENTATION.md`. Each step ends with a check. Do not continue until it passes.

| Step | What | Script | Check |
|---|---|---|---|
| C1 | Get a Power10/Power11 LPAR (IBM Cloud PowerVS **or** on-premises) | `ibmcloud/provision_powervs.sh` | `uname -m` = `ppc64le` |
| C2 | OS packages, SMT4, MMA check | `scripts/os_prep.sh`, `scripts/check_power.sh` | `RESULT: PASS` |
| C3 | Python 3.12 venv from IBM's Power wheels | `scripts/make_venv.sh` | `torch 2.10.0+ppc64le1` printed |
| C4 | Build llama.cpp with Power10 MMA kernels | `scripts/build_llama_server.sh` | `[ OK ] MMA kernels present` |
| C5 | Download Granite 4.2 GGUF and run `llama-server` | `scripts/get_model.sh`, `scripts/run_llama_server.sh` | `curl /health` → `ok` |
| C6 | Build whisper.cpp (Solution 04 only) | `scripts/build_whisper_cpp.sh` | `[ OK ] MMA kernels present` |
| C7 | Run as services (systemd) | `deploy/*.service` | `systemctl status` active after reboot |
| C8 | After fine-tuning: convert to GGUF | `scripts/hf_to_gguf.sh` | new `.gguf` answers through llama-server |

Conventions used everywhere:

* The repo is cloned to **`/opt/ibm-power-ai-solutions`** on the LPAR.
* Models live in **`/opt/models`**. Hugging Face cache: `HF_HOME` (default `~/.cache/huggingface`).
* OS: **RHEL 9.4+ ppc64le** (Ubuntu 24.04 ppc64el works with `apt` equivalents). Python **3.12**.
* One LLM server (`llama-server` on port 8080, OpenAI-compatible API) is shared by all solutions on an LPAR.

---

## C1 — Get a Power LPAR

### Option A: IBM Cloud, Power Virtual Server (PowerVS)

1. In the IBM Cloud console create a **Power Virtual Server workspace** in the data center closest to
   your users that your data-residency rules allow (`ibmcloud pi datacenter list`). There is no IBM Cloud
   region in Iraq. For data that must stay in Iraq, use **Option B**.
2. Install the CLI on your laptop: <https://cloud.ibm.com/docs/cli> then:

```bash
ibmcloud login --sso
ibmcloud plugin install power-iaas -f
ibmcloud pi workspace list                         # copy the CRN
ibmcloud pi workspace target <WORKSPACE_CRN>
ibmcloud pi image list-catalog | grep -i rhel      # pick a stock RHEL 9 image
ibmcloud pi image create <CATALOG_IMAGE_ID>        # copy it into the workspace
ibmcloud pi subnet list
ibmcloud pi ssh-key create mykey --key "$(cat ~/.ssh/id_rsa.pub)"
```

3. Create the LPAR with the size the solution asks for:

```bash
export WORKSPACE_CRN=... IMAGE_ID=... SUBNET_ID=... SSH_KEY_NAME=mykey
export SYS_TYPE=s1122          # Power11. Use s1022 for Power10 if s1122 is not offered in your zone
./common/ibmcloud/provision_powervs.sh ai-lpar-01 medium
```

| Size | Cores (dedicated) | RAM | Good for |
|---|---:|---:|---|
| small | 4 | 32 GB | Granite 4.2-3B, Solutions 05, 06, 07 |
| medium | 8 | 64 GB | Granite 4.2-8B, Solutions 01–04, 08, 09 |
| large | 16 | 128 GB | Granite 4.2-30B, or several solutions on one LPAR |

Flag names change between `power-iaas` plugin versions; `ibmcloud pi instance create --help` is the truth.
**Dedicated** cores give stable performance; **shared** is cheaper for development.

### Option B: on-premises IBM Power (recommended for Iraqi enterprises)

Ask your Power administrator (HMC or PowerVM NovaLink) for an LPAR with RHEL 9.4+, the cores/RAM above,
**dedicated processors**, and Power10/Power11 **native** processor compatibility mode (Power9 compatibility
mode hides MMA). Entry servers that fit this repo: **Power S1012** (edge/branch, 1 socket),
**Power S1022/S1024** and the Power11 equivalents. Everything in this repo runs fully offline after models are
downloaded, so an air-gapped LPAR works: download models on a connected machine, copy `/opt/models`.

**Check:** `ssh` in and run `uname -m && grep PRETTY_NAME /etc/os-release` → `ppc64le` and RHEL 9.

---

## C2 — OS preparation and MMA check

```bash
sudo dnf install -y git
sudo git clone https://github.com/<your-org>/ibm-power-ai-solutions /opt/ibm-power-ai-solutions
sudo chown -R "$USER" /opt/ibm-power-ai-solutions && cd /opt/ibm-power-ai-solutions
sudo common/scripts/os_prep.sh --smt4
common/scripts/check_power.sh
```

**MMA** (Matrix-Multiply Assist) is the set of matrix units inside every Power10/Power11 core. The check
reads the kernel's `AT_HWCAP2` flags; `mma` must be there. **SMT4** is recommended because the 4 MMA
units are shared by the hardware threads of a core.

**Check:** last line `RESULT: PASS`.

---

## C3 — Python environment from IBM's Power wheels

Every solution has its own `requirements.txt` whose first line is IBM's wheel index:

```
--extra-index-url https://wheels.developerfirst.ibm.com/ppc64le/linux-v2026.03.31/+simple/
--prefer-binary
```

* That is the **dated snapshot** the [ppc64le/pyeco](https://github.com/ppc64le/pyeco) Python 3.12 examples are
  tested against (torch `2.10.0+ppc64le1`, numpy `2.4.1+ppc64le1`, ...). Packages with a `+ppc64le` suffix come
  from IBM, pure-Python packages (transformers, peft, ...) from PyPI.
* Keep **`/+simple/`**: the index is a devpi server; without it pip receives an HTML page and finds nothing.
* The rolling index `https://wheels.developerfirst.ibm.com/ppc64le/linux/+simple/` has newer builds
  (torch 2.12+, vllm 0.30 CPU, ...). Move to it deliberately, one solution at a time, and re-run its checks.
* **RocketCE (conda)**: `conda install -c rocketce pytorch-base` gives Power-optimized conda packages, but the
  channel is older (pytorch-base 2.1.2, transformers 4.38, onnxruntime 1.16.3 as of 2026-11-15): too old for
  Granite 4.2. Use it only for legacy conda environments.

```bash
cd solutions/02-enterprise-rag
../../common/scripts/make_venv.sh            # creates ./venv
source venv/bin/activate
```

**Check:** the script prints `torch 2.10.0+ppc64le1` (solutions that use torch).

---

## C4 — llama.cpp with Power10 MMA kernels

```bash
sudo mkdir -p /opt/llama.cpp && sudo chown "$USER" /opt/llama.cpp
common/scripts/build_llama_server.sh
```

The build uses `-DGGML_NATIVE=OFF -DGGML_CPU_POWERPC_CPUTYPE=power10` so GCC emits Power10 code with the
MMA builtins; ggml then uses its MMA matrix kernels for **Q4_0, Q8_0, BF16 and F32** weights. That is why this
repo uses `Q4_0`/`Q8_0` GGUF files rather than the K-quants (`Q4_K_M`).

**Check:** `[ OK ] MMA kernels present` (the script counts `xv*ger*` MMA instructions with `objdump`).

---

## C5 — Granite 4.2 on llama-server

```bash
source solutions/<any>/venv/bin/activate                         # for the `hf` CLI
sudo mkdir -p /opt/models && sudo chown "$USER" /opt/models
common/scripts/get_model.sh ibm-granite/granite-4.2-8b-GGUF granite-4.2-8b-Q4_0.gguf   # ~5 GB
common/scripts/run_llama_server.sh /opt/models/granite-4.2-8b-Q4_0.gguf 8080
```

In another shell:

```bash
curl -s localhost:8080/health
curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages":[{"role":"user","content":"ما هي عاصمة العراق؟ أجب بجملة واحدة."}],
  "chat_template_kwargs":{"enable_thinking":false}, "max_tokens":64}' | jq -r '.choices[0].message.content'
```

| Model file | RAM | Use |
|---|---:|---|
| `granite-4.2-3b-Q4_0.gguf` / `Q8_0` | ~3 / ~4.5 GB | small LPARs, classification, extraction, high throughput |
| `granite-4.2-8b-Q4_0.gguf` / `Q8_0` | ~5 / ~9 GB | **default**: RAG, agents, summaries |
| `granite-4.2-30b-Q4_0.gguf` | ~18 GB | hardest reasoning, needs 16 cores for usable speed |

Granite 4.2 thinks (`<think>…</think>`) by default. `common/python/llm_client.py` sends
`enable_thinking: false` per request; pass `thinking=True` only where the extra quality is worth the CPU time.
Big LPARs spanning several NUMA nodes: start with `NUMA_NODE=0`. Benchmark with
`/opt/llama.cpp/build/bin/llama-bench -m <gguf> -t <cores>`.

**Check:** `/health` returns `{"status":"ok"}` and the Arabic question gets an Arabic answer.

---

## C6 — whisper.cpp (Solution 04)

```bash
sudo mkdir -p /opt/whisper.cpp && sudo chown "$USER" /opt/whisper.cpp
common/scripts/build_whisper_cpp.sh                 # also downloads ggml-large-v3-turbo-q8_0.bin
```

**Check:** `[ OK ] MMA kernels present` and `/opt/models/ggml-large-v3-turbo-q8_0.bin` exists.

---

## C7 — Run as services

```bash
sudo useradd -r -s /sbin/nologin ai || true
sudo chown -R ai:ai /opt/models
sudo mkdir -p /etc/llama-server
sudo cp common/deploy/granite-8b.env.example /etc/llama-server/granite-8b.env
sudo cp common/deploy/llama-server@.service common/deploy/ai-app@.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now llama-server@granite-8b
sudo systemctl enable --now ai-app@02-enterprise-rag          # any solution folder name
```

The APIs listen on `127.0.0.1`. Publish them through your reverse proxy (nginx/HAProxy with TLS and your SSO)
or an SSH tunnel: `ssh -N -L 8000:127.0.0.1:8000 user@lpar`.

**Check:** `sudo reboot`, then `systemctl is-active llama-server@granite-8b ai-app@02-enterprise-rag` → `active`.

---

## C8 — Fine-tuned model → GGUF

After `common/python/lora_sft.py --merge` (or any solution's fine-tuning script that saves a merged model):

```bash
common/scripts/hf_to_gguf.sh out/lora/merged /opt/models/granite-4.2-3b-mytask Q4_0
sudo cp /etc/llama-server/granite-8b.env /etc/llama-server/mytask.env   # edit GGUF= and PORT=
sudo systemctl enable --now llama-server@mytask
```

**Check:** the solution's evaluation script scores the new endpoint at least as well as the base model.

---

## Optional: GPU for fine-tuning

The default in this repo is **CPU-only**. If a fine-tuning job is too slow on CPU, read
[`COMPLIANCE.md`](../COMPLIANCE.md) first: most current data-center GPUs are export-controlled for Iraq.
The fine-tuning scripts run on CPU by default and accept `--gpu` to use one CUDA device on a separate,
compliant training machine. The trained model always comes back to Power for inference via C8.

---

## Computer-vision layer (Solutions 10 and 12)

`common/python/vision/` holds the pieces both camera solutions share: resilient RTSP readers (`video.py`), motion
gating, the RT-DETRv2 ONNX detector, an IoU tracker, evidence hashing/signing, a SQLite helper with an audit table,
token auth, and the model lifecycle tools:

```bash
common/scripts/make_venv.sh common/requirements-vision-train.txt /opt/venv-vision-train
source /opt/venv-vision-train/bin/activate
python common/python/vision/train_rtdetr.py   --train data/train --val data/val --out out/detector
python common/python/vision/export_rtdetr.py  --model out/detector/best --out models/detector   # dynamo + parity
python common/python/vision/eval_detector.py  --model models/detector/model.onnx --data data/test --gate '...'
python common/python/vision/benchmark.py      --model models/detector/model.onnx --fps 2 --motion-duty 0.3
```

Always export RT-DETR(v2) with `export_rtdetr.py`: the classic TorchScript ONNX exporter gives wrong scores for this
model (measured ~0.1 error); the script uses the dynamo exporter and refuses to finish when parity fails.
