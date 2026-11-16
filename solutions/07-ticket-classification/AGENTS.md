# AGENTS.md — Solution 07 (Ticket classification)

Three parts: **agent brief**, **agentic workflow**, **change log**. Repository rules: [`/AGENTS.md`](../../AGENTS.md).

---

## 1. Agent brief

**Purpose:** classify tickets (category + sentiment) and route them, using a small encoder distilled from Granite 4.2.

**Invariants:**

* Serving (`src/serve.py`) must not import torch: ONNX Runtime only.
* The **test set is human-labelled only**. Never evaluate on LLM labels.
* Labels and routing live in `config.yaml`. Changing labels requires retraining and re-export (Step 6–7).
* `labels.json` is written next to the model and is the source of truth for output order at serving time.
* ONNX export must pass the parity check (`max |torch - onnx| < 1e-3`).

**Map:**

| Change | File |
|---|---|
| Labels, routing | `config.yaml` |
| LLM labelling prompt | `src/llm_label.py` |
| Architecture (heads) | `src/model.py` + `export_onnx.py` output names + `serve.py` |
| Training loop | `finetune/train_classifier.py` |

**Verify:**

```bash
python -m py_compile src/*.py finetune/*.py
python finetune/train_classifier.py && python src/export_onnx.py --int8
python -c "import sys; sys.path.insert(0,'src'); import serve; print(serve.classify(['ماكو انترنت']))"
```

---

## 2. Agentic workflow (runtime)

The classifier routes; a **resolution agent** drafts the first reply for the common, low-risk categories.

```
 ticket ──> /classify ──> category, sentiment, confidence
                 |
   confidence ≥ 0.8 and category in {billing, technical_support, sales, other}
                 |
                 v
 [Resolution agent] Granite 4.2-8B, thinking=off, max 3 tool calls
   tools:
     search_kb(query)               -> Solution 02 /search (product, tariff, troubleshooting docs)
     account_status(customer_ref)   -> read-only: plan, balance, open outages in the customer's area
     draft_reply(text_ar, text_en)  -> stores a draft reply (in the customer's language)
                 |
                 v
 human agent reviews the draft, edits, sends  ──> edits logged as quality data
 network_outage / complaint / negative sentiment ──> straight to a human, no draft
```

**Guardrails:** the agent never sends messages itself; no refunds, plan changes or promises of compensation in
drafts (the prompt forbids it and a keyword filter blocks drafts that mention them); every draft cites KB passages.

**Learning loop:** human edits to drafts and triage corrections both become training data (labels for the classifier,
SFT pairs for a LoRA on Granite via `common/python/lora_sft.py`).

**Implementation path:** Solution 09 `src/agent.py`.

---

## 3. Agent change protocol

1. Change, run every **Verify** command.
2. Docs changed → `python tools/build_docx.py solutions/07-ticket-classification`.
3. Append a row below.

### Agent change log

| Date | Agent / author | Change | Verification result |
|---|---|---|---|
| 2026-11-15 | Claude Code (Opus 5.5) | Initial version: LLM bootstrap labelling, two-head encoder fine-tuning, ONNX export + int8, ONNX Runtime API | Full pipeline run on x86 CPU (torch 2.10.0+cpu, transformers 4.57.6, onnxruntime 1.23.1): training on the 32-ticket demo split (accuracy is meaningless at this size, as documented), ONNX parity max error 1.6e-6, int8 model 278 MB vs 1110 MB fp32, ONNX Runtime classify 87–130 ms per ticket with low-confidence tickets flagged `needs_human`. Not yet run on Power hardware |
