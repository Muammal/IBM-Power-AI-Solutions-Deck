# AGENTS.md — Solution 04 (Speech analytics)

Three parts: **agent brief**, **agentic workflow**, **change log**. Repository rules: [`/AGENTS.md`](../../AGENTS.md).

---

## 1. Agent brief

**Purpose:** call recordings → Arabic transcript → redaction → structured call analytics, on CPU.

**Invariants:**

* ASR runs in **whisper.cpp** (C++), not in Python, for inference. Do not add torch to `requirements.txt`;
  torch belongs only in `requirements-finetune.txt`.
* **Redaction happens before anything is stored or sent to the LLM.** Never log raw transcripts.
  Keep `redact()` covering Arabic-Indic digits.
* `ANALYTICS_SCHEMA` is consumed by dashboards: add fields, do not rename or remove them without a migration note.
* Retention: `scripts/purge_old_calls.sh` must keep working with `config.yaml`.
* `wer.py` normalisation must stay identical between baseline and fine-tuned evaluations.

**Map:**

| Change | File |
|---|---|
| ASR model / language / beam | `config.yaml`, `scripts/run_whisper_server.sh` |
| New analytics field | `ANALYTICS_SCHEMA` in `src/pipeline.py` |
| New personal-data pattern | `_REDACT` in `src/pipeline.py` |
| Fine-tuning recipe | `finetune/finetune_whisper.py` |

**Verify:**

```bash
python -m py_compile src/*.py finetune/*.py eval/*.py
python src/wer.py                                        # prints 0.0 then 0.25
python -c "import sys; sys.path.insert(0,'src'); from pipeline import redact; assert redact('٠٧٧٠١٢٣٤٥٦٧')=='[PHONE]'"
python eval/eval_asr.py data/asr/test                     # WER must not increase
```

**Do not:** commit audio, transcripts (`data/calls/`, `data/asr/`), or models.

---

## 2. Agentic workflow (runtime)

A **call-follow-up agent** reads each analysed call and prepares, but does not send, follow-up actions.

```
 data/calls/<id>.json (redacted transcript + analytics)
        |
        v
 [Triage]  rules, no LLM: escalate==true OR sentiment in {negative, angry} OR resolved==false
        |
        v
 [Follow-up agent]  Granite 4.2-8B, thinking=on, max 4 tool calls
   tools:
     lookup_customer(call_id)           -> CRM record (read-only; phone number never shown to the model)
     search_kb(query)                   -> Solution 02 /search over product & policy documents
     draft_callback_note(text_ar)       -> stores a draft for the supervisor
     create_ticket(category, summary)   -> REQUIRES human approval in the supervisor UI
        |
        v
 supervisor queue: drafts + proposed tickets; one click to approve, edit or reject
```

**Guardrails:** the agent never calls or messages customers; `create_ticket` is a proposal until a human
approves it; the agent sees only redacted transcripts; every action is logged with `call_id`.

**Learning loop:** supervisor edits to transcripts (when they listen to a call) become new ASR training pairs in
`data/asr/train/`; re-run Step 8 quarterly.

**Implementation path:** Solution 09 `src/agent.py` provides the tool-calling loop and an approval gate.

---

## 3. Agent change protocol

1. Change, then run every **Verify** command.
2. Docs changed → `python tools/build_docx.py solutions/04-speech-analytics`.
3. Append a row below.

### Agent change log

| Date | Agent / author | Change | Verification result |
|---|---|---|---|
| 2026-11-15 | Claude Code (Opus 5.5) | Initial version: PyAV decoding, whisper.cpp client, redaction, Granite analytics, Whisper fine-tuning, WER eval | `py_compile` OK; `wer.py` and `redact()` self-tests run on x86; not yet run on Power hardware |
