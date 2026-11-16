# Step 04 — Configuration and choosing a model

## Goal

Understand `config.yaml`, which is **the only file you edit** to change the model or settings.
Choose a model, and give the chatbot your Hugging Face token if the model is gated.

## Files created in this step

| File | Purpose |
|------|---------|
| `config.yaml` | Backend, model ID, dtype, threads, generation settings, system prompts (AR + EN), server |
| `.env.example` | Template for secrets (`HF_TOKEN`) and optional overrides. Copy it to `.env` |
| `docs/steps/04_choose_model.md` | This document |

## The model presets

IDs, licences and gating were checked on Hugging Face while building this template (November 2026).
Check them again on `https://huggingface.co/<id>` before you use a model, because repos can move.
ALLaM already has: `ALLaM-AI/...` now redirects to `humain-ai/...`.

| Preset (`model.id`) | Params | Weights bf16 / fp32 | RAM to plan (bf16 / fp32) | Arabic | Licence / gating |
|---|---:|---:|---:|---|---|
| `meta-llama/Llama-3.2-3B-Instruct` **(default)** | 3.2B | 6.4 / 12.9 GB | 16 / 24 GB | **Not officially supported.** It often answers in Arabic, but test quality and dialect understanding yourself | Llama 3.2 Community Licence. **Gated**: request access on the model page, then set `HF_TOKEN` |
| `Qwen/Qwen2.5-3B-Instruct` | 3.1B | 6.2 / 12.3 GB | 16 / 24 GB | Supported (Arabic is among its 29+ languages) | **Qwen Research Licence: non-commercial only.** Not gated |
| `Qwen/Qwen2.5-7B-Instruct` | 7.6B | 15.2 / 30.5 GB | 24 / 48 GB | Supported, noticeably better than 3B | Apache-2.0. Not gated |
| `humain-ai/ALLaM-7B-Instruct-preview` | 7B | ~14 / ~28 GB | 24 / 48 GB | **Arabic-first** (trained for Arabic and English) | Apache-2.0. Not gated. Uses a SentencePiece tokenizer (`sentencepiece` is in requirements.txt) |

"RAM to plan" = weights + KV cache + Python/OS headroom for one user. Disk use is roughly the bf16 size,
because the files on the Hub are stored in bf16.

### Which one should I start with?

* **Learning the template / smallest box:** `Qwen/Qwen2.5-3B-Instruct` (no token needed, good Arabic) if
  non-commercial use is fine. Otherwise Llama-3.2-3B.
* **Best Arabic quality on CPU:** `humain-ai/ALLaM-7B-Instruct-preview` or `Qwen/Qwen2.5-7B-Instruct`
  on a 64 GB LPAR.
* **Less RAM / more speed:** any of them as a GGUF file with the llama.cpp backend (Step 09).

## Settings explained

| Key | Meaning |
|---|---|
| `backend` | `transformers` (PyTorch, safetensors) or `llamacpp` (GGUF, Step 09) |
| `model.id` | Hugging Face repo ID. Exactly one active line |
| `model.revision` | Branch/tag/commit. Pin a commit hash in production so the model cannot change under you |
| `model.dtype` | `bfloat16` = half the memory and uses Power10 MMA bf16 instructions. `float32` = reference accuracy, 2× RAM |
| `runtime.threads` | PyTorch / OpenBLAS / llama.cpp threads. `0` = **auto: one per physical core** (counted from `/sys`; PyTorch alone would use every SMT thread on POWER). Tune it in Step 08 |
| `generation.*` | `max_new_tokens`, `temperature` (0 = deterministic), `top_p`, `repetition_penalty`, `max_history_turns` |
| `system_prompt.en` / `.ar` | Chosen from the script of the user's latest message (Arabic letters → `ar`) |
| `server.host` / `port` | Web UI address. Default `127.0.0.1:7860` (local only) |

## Commands to run (copy-paste ready)

```bash
cd /opt/power-chatbot
source venv/bin/activate
cp -n .env.example .env
vi .env             # paste your token: HF_TOKEN=hf_xxx   (needed for Llama)
chmod 600 .env
```

To get a token: huggingface.co → Settings → Access Tokens → *New token* (type **Read**).
For Llama, also open https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct and accept the licence.
Access is approved manually, usually within minutes to hours.

## Expected output

No output yet. You only edited two text files.

## How to verify

This command parses the YAML, checks that the token can see the model, and prints the model's size and licence.
It downloads nothing big:

```bash
python - <<'EOF'
import os, llm
from huggingface_hub import model_info
cfg = llm.load_config()
info = model_info(cfg["model"]["id"], token=os.getenv("HF_TOKEN") or None, files_metadata=True)
size = sum(s.size or 0 for s in info.siblings if s.rfilename.endswith(".safetensors"))
print("backend:", cfg["backend"], "| model:", info.id, "| gated:", info.gated,
      "| licence:", (info.card_data or {}).get("license"), "| weights: %.1f GB" % (size / 1e9))
EOF
```

**Passes** when it prints a line like:

```
backend: transformers | model: meta-llama/Llama-3.2-3B-Instruct | gated: manual | licence: llama3.2 | weights: 6.4 GB
```

## Common errors and fixes

| Symptom | Cause | Fix |
|---------|-------|-----|
| `GatedRepoError` / `401` / `403` | No token, or licence not accepted yet | Put `HF_TOKEN` in `.env`; accept the licence on the model page and wait for approval |
| `RepositoryNotFoundError` | Typo, or the repo moved | Search huggingface.co and correct `model.id` |
| `yaml.scanner.ScannerError` | Bad indentation in `config.yaml` | YAML needs spaces, never tabs. Keep the 2-space indent |
| Two `id:` lines active | Forgot to comment one out | YAML keeps the **last** one silently. Keep exactly one active |
| Arabic in config shows as `???` | File saved in a non-UTF-8 encoding | Save `config.yaml` as UTF-8 |
