# AGENTS.md — Solution 03 (Document intelligence)

Three parts: **agent brief** (for AI coding agents), **agentic workflow** (runtime design), **change log**.
Repository-wide rules: [`/AGENTS.md`](../../AGENTS.md).

---

## 1. Agent brief

**Purpose:** scanned/PDF business documents (Arabic/English) → schema-valid JSON → rule validation → approve or review.

**Invariants:**

* CPU-only on ppc64le. The core `requirements.txt` must stay torch-free and install from the dated IBM index.
  Docling stays optional (`requirements-docling.txt`); `convert.py` must keep working without it.
* All LLM output goes through a JSON schema (`schemas/*.json`). Never parse free-text LLM output with regex.
* Validation rules are deterministic Python (`validate_*` in `extract.py`). Do not ask the LLM to "check the maths".
* `normalize_digits` runs on all text before the LLM. Keep Arabic-Indic and Extended Arabic-Indic digits mapped.
* Never auto-approve a document whose validation returned problems.

**Map:**

| Change | File |
|---|---|
| New document type | `schemas/<type>.json` + `validate_<type>()` in `src/extract.py` + gold files |
| OCR engine/options | `src/convert.py`, `config.yaml: converter` |
| Approval rules | `src/extract.py: validate_invoice()` |
| API | `src/app.py` |

**Verify:**

```bash
python -m py_compile src/*.py finetune/*.py eval/*.py
python src/extract.py data/samples/invoice_001.md          # status auto_approved, total 1080000
python eval/eval_extraction.py --docs data/samples --gold data/gold   # no field may regress
```

**Do not:** commit real invoices, `data/reviewed/`, or `out/`; send documents to external APIs.

---

## 2. Agentic workflow (runtime)

The pipeline is deterministic until validation fails. That is where an agent helps: instead of sending every
`needs_review` document to a person, a **review agent** tries to fix it with tools, and escalates only what it cannot.

```
 needs_review document + problems
        |
        v
 [Review agent] Granite 4.2-8B, thinking=on, max 5 tool calls
   tools:
     reocr_page(page, dpi, psm)       -> re-run Tesseract on one page with other settings
     get_vendor_profile(tax_id)       -> past approved invoices from this vendor (layout hints, usual currency)
     recompute_totals(line_items)     -> deterministic sums (the agent never does arithmetic itself)
     propose_fix(field, value, reason)-> records a change; does NOT approve
        |
        v
 validate again (same Python rules)
   |-- passes and every change has a reason and source  -> "agent_fixed" (human spot-checks 10%)
   |-- fails or confidence low                           -> human review queue, with the agent's notes
```

Tool schema example:

```json
{"type": "function", "function": {
  "name": "propose_fix",
  "description": "Propose a corrected value for one field. Must quote the evidence from the document text.",
  "parameters": {"type": "object", "properties": {
     "field": {"type": "string"}, "value": {}, "evidence": {"type": "string"}},
   "required": ["field", "value", "evidence"]}}}
```

**Guardrails:** the agent cannot approve, cannot write to the ERP, and cannot change `total` without evidence
quoting the document. Humans own approval. All proposals are logged with the document hash.

**Learning loop:** human-corrected documents → `data/reviewed/` → `finetune/build_sft.py` → LoRA (Step 8). The agent's
accepted fixes are also good training examples.

**Implementation path:** use the tool-calling loop in Solution 09 (`src/agent.py`): register the four tools as
Python functions and pass the `needs_review` JSON as the task.

---

## 3. Agent change protocol

1. Make the change, run all **Verify** commands.
2. Regenerate Word files when docs change: `python tools/build_docx.py solutions/03-document-intelligence`.
3. Append a row below (newest last).

### Agent change log

| Date | Agent / author | Change | Verification result |
|---|---|---|---|
| 2026-11-15 | Claude Code (Opus 5.5) | Initial version: convert (basic/Docling), schema extraction, validation, batch, API, LoRA data builder, eval | `py_compile` OK; validation rules unit-checked (gold file: no problems; bad date/total: both caught); digit normalisation checked; core requirements (18 packages) and Docling extra (106 packages, IBM rolling index incl. `docling-parse 7.23.0+ppc64le`) resolve for ppc64le/cp312. OCR and LLM extraction not run in this session; not yet run on Power hardware |
| 2026-11-21 | Claude Code (Opus 5.5) | `src/extract.py` prompt: copy amounts exactly as printed (never recompute) and define subtotal. Granite 4.2-8B had silently corrected a wrong line amount (4 × 95,000 printed as 390,000 → returned 380,000), so validation auto-approved a faulty invoice | Power10 (PowerVS s1022, 2 cores, Granite 4.2-8B Q4_0 on llama-server, Tesseract 5.5 + poppler 26.11 from conda-forge): `eval_extraction.py` 6/7 → 7/7 fields (subtotal 0% → 100%), still 1/1 auto-approved; fictional invoice with wrong line 2: auto_approved → needs_review ("line 2: quantity x unit_price != amount"); correct invoice still auto_approved; ~60–70 s per invoice on 2 cores. Phone-scan PNG with bordered table: Tesseract loses the quantity column, result needs_review (safe) |
