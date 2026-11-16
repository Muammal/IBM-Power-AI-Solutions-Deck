# Step 11 — How to switch models (the main teaching document)

## Goal

Learn the one skill this template exists to teach: **changing the model by editing `config.yaml` only.**
No Python file changes, ever. You will go from Llama → Qwen → ALLaM, then try a GGUF file, and learn
what to re-test after each change.

## Files created in this step

| File | Purpose |
|------|---------|
| `docs/steps/11_swap_models.md` | This document (no new code: that is the point) |

## Why no code changes are needed

Everything model-specific is handled by the model's own files or by `config.yaml`:

| What differs between models | Where it comes from |
|---|---|
| Prompt format (`<\|im_start\|>`, `[INST]`, `<\|eot_id\|>`, ...) | the model's **chat template**, applied by `tokenizer.apply_chat_template` (or the GGUF metadata in llama.cpp) |
| Architecture (Llama, Qwen2, ...) | `config.json` in the repo; `AutoModelForCausalLM` picks the right class |
| Tokenizer | `AutoTokenizer` reads `tokenizer.json` or `tokenizer.model` |
| Stop tokens | `generation_config.json` in the repo |
| Model ID, dtype, threads, system prompt, sampling | `config.yaml` |

## The swap procedure (same every time)

1. **Stop** the running app (`Ctrl+C`, or `sudo systemctl stop chatbot`).
2. **Edit `config.yaml`**: comment out the old `id:` line and uncomment (or add) the new one. Exactly one active `id:`.
3. **Check RAM** against the table in Step 04 and adjust `dtype` if needed.
4. **Download** the new model (first run does it automatically; for the offline service see Step 10, Option A step 3).
5. **Re-test** with the checklist at the end of this document.

---

## Example 1 — Llama 3.2 3B → Qwen2.5 3B

Reason: better Arabic support out of the box, and no gated access.

**Before**

```yaml
model:
  id: meta-llama/Llama-3.2-3B-Instruct
  # id: Qwen/Qwen2.5-3B-Instruct
  # id: Qwen/Qwen2.5-7B-Instruct
  # id: humain-ai/ALLaM-7B-Instruct-preview
  revision: main
  dtype: bfloat16
```

**After**

```yaml
model:
  # id: meta-llama/Llama-3.2-3B-Instruct
  id: Qwen/Qwen2.5-3B-Instruct
  # id: Qwen/Qwen2.5-7B-Instruct
  # id: humain-ai/ALLaM-7B-Instruct-preview
  revision: main
  dtype: bfloat16
```

```bash
python cli.py --selftest
```

What changed behind the scenes (you did nothing for it):
* Prompt format switched from Llama's `<|start_header_id|>user<|end_header_id|> ... <|eot_id|>` to Qwen's
  `<|im_start|>user ... <|im_end|>`.
* `HF_TOKEN` is no longer needed (Qwen is not gated). Leaving it set does no harm.

Watch out:
* **Licence:** Qwen2.5-**3B** uses the *Qwen Research Licence* (**non-commercial**). For a commercial service,
  use Qwen2.5-**7B** (Apache-2.0) or ALLaM (Apache-2.0).
* Same size class, so RAM and speed should be similar. Still re-run the benchmark (Step 08).

## Example 2 — Qwen2.5 3B → ALLaM 7B (Arabic-first)

Reason: the best Arabic, including dialect understanding, among the presets.

**Before**

```yaml
model:
  id: Qwen/Qwen2.5-3B-Instruct
  # id: humain-ai/ALLaM-7B-Instruct-preview
  revision: main
  dtype: bfloat16
runtime:
  threads: 16
generation:
  max_history_turns: 6
```

**After**

```yaml
model:
  # id: Qwen/Qwen2.5-3B-Instruct
  id: humain-ai/ALLaM-7B-Instruct-preview
  revision: main
  dtype: bfloat16          # 7B: ~14 GB in bf16. float32 would need ~28 GB + headroom
runtime:
  threads: 16              # re-tune: the best value for 7B can differ from 3B
generation:
  max_history_turns: 4     # 7B is ~2x slower per token, so keep the prompt shorter
```

```bash
free -g                       # make sure ~24 GB are available
python cli.py --selftest
python scripts/benchmark.py --dtypes bfloat16 --threads 8 16 32
```

What changed behind the scenes:
* ALLaM ships a **SentencePiece** tokenizer (`tokenizer.model`, no `tokenizer.json`). The `sentencepiece` and
  `protobuf` packages in `requirements.txt` exist for this.
* ALLaM's template is **Llama-2 style** (`[INST] <<SYS>> ... <</SYS>> ... [/INST]`) and **raises an error unless turns
  alternate user/assistant**. `llm.build_messages` already guarantees that, and the web UI drops an unanswered question.
* The repo ID moved from `ALLaM-AI/ALLaM-7B-Instruct-preview` to `humain-ai/ALLaM-7B-Instruct-preview`. The old ID still
  redirects, but use the new one. This is why the presets say "confirm the exact ID on Hugging Face".

## Example 3 — Same model, smaller and faster: switch to a GGUF file

Reason: less RAM, often higher tokens/s with llama.cpp's MMA kernels (Step 09 must be done first).

**Before**

```yaml
backend: transformers
llamacpp:
  gguf_path: models/Llama-3.2-3B-Instruct-Q4_0.gguf
```

**After**

```yaml
backend: llamacpp
llamacpp:
  gguf_path: models/qwen2.5-3b-instruct-q8_0.gguf    # downloaded by scripts/09_setup_llamacpp.sh
  n_ctx: 4096
```

```bash
./scripts/09_setup_llamacpp.sh Qwen/Qwen2.5-3B-Instruct-GGUF qwen2.5-3b-instruct-q8_0.gguf
python cli.py --selftest
```

Notes:
* With `backend: llamacpp`, the `model:` section is ignored. Only `gguf_path` matters.
* The prompt format now comes from the chat template embedded in the GGUF file.
* Q8_0 is the safe choice for Arabic quality. Try Q4_0 if you need more speed or less RAM, and re-run the selftest.

## Example 4 — Keep several configs instead of editing one

Editing one file is simplest. When you compare models often, keep one file per model and choose with an
environment variable. Still no code change:

```bash
mkdir -p configs
cp config.yaml configs/llama-3b.yaml
cp config.yaml configs/allam-7b.yaml   # then edit model.id inside it
CHATBOT_CONFIG=configs/allam-7b.yaml python cli.py --selftest
CHATBOT_CONFIG=configs/allam-7b.yaml python app.py
python cli.py --config configs/llama-3b.yaml --selftest
```

## Example 5 — A model that is not in the presets

Before you put any new `id:` in the config, check these on its Hugging Face page:

| Check | Why |
|---|---|
| It is an **Instruct/Chat** model | `llm.py` stops with "has no chat template" otherwise |
| `tokenizer_config.json` contains `chat_template` | Needed for `apply_chat_template` |
| Licence and gating fit your use | Some are non-commercial or need approval |
| Size × 2 bytes (bf16) fits in RAM with headroom | See Step 04 |
| It works **without** `trust_remote_code` | Keep `trust_remote_code: false`; models that need custom code run downloaded Python |
| Supported by transformers 4.57.6 | Very new architectures may need a newer transformers (a requirements change, then re-run Step 03) |
| Arabic is a listed language | Otherwise test Arabic carefully, as with Llama |

## Pin the exact version for production

`revision: main` follows the repo, so the model can change after an update. Pin the commit you tested:

```bash
python -c "from huggingface_hub import model_info; print(model_info('Qwen/Qwen2.5-7B-Instruct').sha)"
```

```yaml
model:
  id: Qwen/Qwen2.5-7B-Instruct
  revision: <the 40-character sha printed above>
```

## What to re-test after every swap

| # | Test | Command / action | Pass when |
|---|---|---|---|
| 1 | Loads, and fits in RAM | `python cli.py --once "hi"` + `free -g` in another terminal | Answers; no `Killed` |
| 2 | Arabic + English, incl. Iraqi dialect | `python cli.py --selftest` | `SELFTEST PASSED`, exit code 0 |
| 3 | Answers stay in the user's language | Ask 3–5 of your real questions in each language | No language switching mid-answer |
| 4 | Multi-turn conversation | Interactive `cli.py`: ask a follow-up ("and its population?") | Uses the previous turn |
| 5 | Speed | `python scripts/benchmark.py ...` | TTFT and tok/s acceptable for your users |
| 6 | Web UI | Step 07 checks (RTL, streaming, Stop) | All pass |
| 7 | Service | Pre-download into the service cache, `systemctl restart chatbot`, `/health` | `"status":"ok"` |
| 8 | Licence | Read the model card | Fits your use |

## Commands to run (copy-paste ready)

The full Llama → Qwen swap from Example 1, without opening an editor:

```bash
cd /opt/power-chatbot && source venv/bin/activate
cp config.yaml config.yaml.bak                                     # easy rollback
sed -i -e 's|^  id: meta-llama/Llama-3.2-3B-Instruct|  # id: meta-llama/Llama-3.2-3B-Instruct|' \
       -e 's|^  # id: Qwen/Qwen2.5-3B-Instruct|  id: Qwen/Qwen2.5-3B-Instruct|' config.yaml
grep -E '^\s+#? ?id:' config.yaml                                  # exactly one line without "#"
python cli.py --selftest
```

Roll back with `mv config.yaml.bak config.yaml`.

## Expected output

```
  # id: meta-llama/Llama-3.2-3B-Instruct
  id: Qwen/Qwen2.5-3B-Instruct
  # id: Qwen/Qwen2.5-7B-Instruct
  # id: humain-ai/ALLaM-7B-Instruct-preview
Loading model ...
Ready in 10.4s: transformers | Qwen/Qwen2.5-3B-Instruct | bfloat16 | 16 threads
...
SELFTEST PASSED
```

## How to verify

Do Example 1 (or any swap you can run) and check that **only `config.yaml` changed**:

```bash
git diff --stat            # if the project is in git: only config.yaml is listed
python cli.py --selftest   # SELFTEST PASSED with the new model in the "Ready" line
```

> Done while building this template: the same unchanged code ran `Qwen2.5-0.5B-Instruct` → `Qwen2.5-1.5B-Instruct`
> (transformers) → `qwen2.5-0.5b-instruct-q4_0.gguf` (llama.cpp). Each time only `model.id`, or `backend` and
> `gguf_path`, differed between the config files.

## Common errors and fixes

| Symptom | Cause | Fix |
|---------|-------|-----|
| The old model still answers | App not restarted, or two `id:` lines active | Restart; keep exactly one active `id:` |
| `GatedRepoError` after switching back to Llama | Token missing in this shell/service | Check `.env`; for the service, `.env` must be readable by the `chatbot` user |
| `TemplateError: Conversation roles must alternate` | Strict template (ALLaM) and a broken history | Start a new chat; don't send two user messages in a row through the API |
| `KeyError` / `does not recognize this architecture` | Model newer than transformers 4.57.6 | Pick another model, or upgrade transformers deliberately (and re-run Steps 03 and 06) |
| Model answers in English to Arabic questions | Weak Arabic in that model | Use Qwen2.5-7B or ALLaM; optionally strengthen `system_prompt.ar` |
| Service fails: `LocalEntryNotFoundError` | New model not in the offline cache | Pre-download it for the service user (Step 10) |
