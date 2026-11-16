# AGENTS.md — Solution 01 (Arabic/English chatbot)

Three parts: **agent brief**, **agentic workflow**, **change log**. Repository rules: [`/AGENTS.md`](../../AGENTS.md).

---

## 1. Agent brief

**Purpose:** a teaching-quality chatbot template (FastAPI + one HTML page) running an LLM on CPU on IBM Power,
answering in the user's language (Arabic incl. Iraqi dialect, or English).

**Invariants:**

* **Changing the model must never need a code change**: everything model-specific goes in `config.yaml`
  (this is the template's main lesson, documented in `docs/steps/11_swap_models.md`).
* `requirements.txt` stays pinned to IBM's dated wheel snapshot with `/+simple/` and installs with
  `--only-binary=:all:`. `llama-cpp-python` is compiled on the LPAR (Step 09), never pinned in requirements.
* Thread count = physical cores (`physical_cores()`), set before torch is imported.
* Prompts are built only with the model's chat template (`apply_chat_template`), never by hand.
* The stop event must stop generation within about a second after a client disconnects.
* Keep files small and commented: this is learning material.

**Map:**

| Change | File |
|---|---|
| Model, dtype, threads, prompts, template variables | `config.yaml` |
| Model loading / generation | `llm.py` |
| Web API | `app.py`; page: `static/index.html` |
| Terminal + selftest | `cli.py` |
| Step documentation | `docs/steps/NN_*.md` (keep the Goal / Files / Commands / Expected / Verify / Errors layout) |

**Verify:**

```bash
python -m py_compile llm.py cli.py app.py scripts/benchmark.py
bash -n scripts/*.sh
python cli.py --selftest           # MSA + Iraqi + English; must print SELFTEST PASSED
```

---

## 2. Agentic workflow (runtime)

The chatbot is single-turn Q&A. To make it an **assistant that can act**, route each message through a small
router and reuse the shared agent loop (Solution 09 `src/agent.py`).

```
 user message (AR/EN)
        |
        v
 [Router] Granite 4.2-3B, json_schema {intent: chat | company_question | action_request}
        |-- chat ─────────────> current chatbot (llm.generate)
        |-- company_question ──> tool search_documents (Solution 02 /search) → answer with citations
        |-- action_request ────> Agent.run(task) with tools such as
                                    lookup_leave_balance(employee_id)      read-only
                                    submit_leave_request(dates, reason)    requires approval
```

**Guardrails:** the user's identity comes from SSO, never from the chat text; write actions need explicit
confirmation in the UI; tool outputs are data, not instructions.

---

## 3. Agent change protocol

1. Change, run every **Verify** command.
2. Docs changed → `python tools/build_docx.py solutions/01-arabic-chatbot`.
3. Append a row below.

### Agent change log

| Date | Agent / author | Change | Verification result |
|---|---|---|---|
| (earlier) | Claude Code | Original power-chatbot template, Steps 01–11 | See "What was verified" in README.md |
| 2026-11-15 | Claude Code (Opus 5.5) | Moved into the solutions repo as Solution 01; default model → `ibm-granite/granite-4.2-3b`; `model.chat_template_kwargs` passed to `apply_chat_template` (Granite `enable_thinking: false`); added IMPLEMENTATION/RESOURCES/AGENTS | Granite 4.2 tokenizer template checked on x86 (transformers 4.57.6): `enable_thinking=False` gives `<think></think>` prefill; `py_compile` OK. Full selftest with Granite 4.2-3B not run in this session |
