# Solution 01 — Step-by-step implementation

This solution already has a detailed 11-step guide in [`docs/steps/`](docs/steps/). This page maps it onto
the two deployment targets and adds fine-tuning, so it reads like the other solutions in the repo.
Each step ends with a **Check** (the original step documents contain the exact expected output).

| # | Step | IBM Cloud (PowerVS) | On-premises Power | Check |
|---|---|---|---|---|
| 1 | [Provision the LPAR](docs/steps/01_provision_power.md) | Option A: `ibmcloud pi instance create` (`s1122` Power11 / `s1022` Power10) or `common/ibmcloud/provision_powervs.sh` | Option B: LPAR from HMC, dedicated cores, native Power10/11 mode | `uname -m` = `ppc64le` |
| 2 | [OS + MMA](docs/steps/02_verify_mma.md) | same | same | `scripts/check_power.sh` → `RESULT: PASS` |
| 3 | [Python env from IBM wheels](docs/steps/03_python_env.md) | same | air-gapped: mirror the IBM index with `pip download` first | torch `2.10.0+ppc64le1` |
| 4 | [Choose a model](docs/steps/04_choose_model.md) | default `ibm-granite/granite-4.2-3b` (no token needed) | copy `~/.cache/huggingface` in if offline | model info line |
| 5 | [Model layer](docs/steps/05_llm_module.md) | | | EN + AR answers |
| 6 | [CLI smoke test](docs/steps/06_cli_test.md) | | | `SELFTEST PASSED` |
| 7 | [Web UI](docs/steps/07_web_ui.md) | SSH tunnel from your laptop | internal reverse proxy + SSO | RTL bubbles stream |
| 8 | [Benchmark](docs/steps/08_benchmark_tuning.md) | dedicated cores for stable numbers | | CSV rows |
| 9 | [llama.cpp backend](docs/steps/09_llamacpp_backend.md) | | | same selftest |
| 10 | [Run as a service](docs/steps/10_deploy.md) | | | `/health` after reboot |
| 11 | [Switch models](docs/steps/11_swap_models.md) | | | selftest with new model |

## Granite 4.2 notes (new default)

* `config.yaml` passes `chat_template_kwargs: {enable_thinking: false}` to `apply_chat_template`, so the
  model answers directly instead of writing `<think>…</think>` first (verified: the prompt then ends with
  `<|im_start|>assistant\n<think></think>`).
* Want reasoning for hard questions? Set `enable_thinking: true` and raise `max_new_tokens` (thinking tokens count).
* **llama.cpp backend (Step 09):** `llama-cpp-python` does not pass template variables, so a Granite 4.2 GGUF will
  think by default. For GGUF + no thinking use the shared `llama-server` from
  [`common/README.md`](../../common/README.md) C5 (it sends `enable_thinking: false` per request), or keep the
  transformers backend.

## Step 12 — Fine-tune the chatbot on your organisation's style (optional, CPU)

1. Collect 300–2,000 real conversations (question + the answer your best agent would give), Arabic and English,
   personal data removed, as JSONL: `{"messages":[{"role":"user",...},{"role":"assistant",...}]}`.
2. Train a LoRA adapter on CPU with the shared script:

```bash
../../common/scripts/make_venv.sh ../../common/requirements-finetune.txt /opt/venv-finetune
source /opt/venv-finetune/bin/activate
python ../../common/python/lora_sft.py --data data/chat_train.jsonl --eval-data data/chat_eval.jsonl \
    --model ibm-granite/granite-4.2-3b --out out/lora-chat --epochs 2 --merge
```

3. Point `model.id` in `config.yaml` at `out/lora-chat/merged` (transformers backend), or convert it with
   `../../common/scripts/hf_to_gguf.sh out/lora-chat/merged /opt/models/granite-4.2-3b-chat Q4_0` for llama.cpp.
4. Run `python cli.py --selftest` and a blind A/B review with staff before switching.

GPU option: `--gpu` on a separate machine whose GPU passes [`COMPLIANCE.md`](../../COMPLIANCE.md).

**Check:** selftest passes and reviewers prefer the fine-tuned answers.
