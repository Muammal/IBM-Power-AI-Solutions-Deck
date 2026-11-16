# AGENTS.md — Solution 08 (Translation AR / KU / EN)

Three parts: **agent brief**, **agentic workflow**, **change log**. Repository rules: [`/AGENTS.md`](../../AGENTS.md).

---

## 1. Agent brief

**Purpose:** translate between Arabic, Kurdish (Sorani `ckb`, Kurmanji `ku`) and English on CPU, with glossary
enforcement and LoRA domain adaptation.

**Invariants:**

* Default model stays commercially licensed (MADLAD-400, Apache-2.0). Never switch to NLLB (non-commercial).
* MADLAD target tag format is `<2xx>` at the start of the input. Language codes live in `config.yaml: languages`.
* Glossary checks run on every translation; post-edit may only fix missing terms (prompt says "change nothing else").
* Granite post-edit is limited to `ar`/`en` targets.
* Evaluation data (`parallel_eval.tsv`) must never be used for training.

**Map:**

| Change | File |
|---|---|
| Model, beams, adapter | `config.yaml` |
| Sentence splitting, glossary logic, post-edit | `src/translate.py` |
| Training | `finetune/lora_madlad.py` |
| Metric | `eval/eval_chrf.py` |

**Verify:**

```bash
python -m py_compile src/*.py finetune/*.py eval/*.py
python src/translate.py --to en "تم استلام طلبك وهو قيد المراجعة"
python eval/eval_chrf.py data/parallel_eval.tsv        # chrF must not drop in any direction
```

---

## 2. Agentic workflow (runtime)

A **document translation agent** handles whole documents with review loops instead of single strings.

```
 document (.docx / .pdf via Solution 03)  + target language
        |
        v
 [Planner] split into segments, detect tables, headings, numbers, names
        |
        v
 [Translator] tool: translate(segment, target)            -> MADLAD (+ adapter)
        |
        v
 [Checker] deterministic: numbers/dates preserved? glossary terms present? segment length ratio sane?
        |-- issues --> [Fixer] Granite 4.2 (ar/en targets) or re-translate with num_beams=8 (ckb/ku)
        v
 [Assembler] rebuild the document, add a "machine translated, review required" footer
        |
        v
 human reviewer (native speaker) ──> corrected segments ──> data/parallel_train.tsv (LoRA, Step 6)
```

**Guardrails:** numbers, dates, IBANs and names are compared between source and target by code, not by the LLM;
nothing is published without human review; reviewer corrections are the training data.

**Implementation path:** Solution 09 `src/agent.py` for the loop; `translate` and the checks are plain Python tools.

---

## 3. Agent change protocol

1. Change, run every **Verify** command.
2. Docs changed → `python tools/build_docx.py solutions/08-translation-ar-ku-en`.
3. Append a row below.

### Agent change log

| Date | Agent / author | Change | Verification result |
|---|---|---|---|
| 2026-11-15 | Claude Code (Opus 5.5) | Initial version: MADLAD translation, glossary check, Granite post-edit, LoRA fine-tuning, chrF eval | `py_compile` OK; chrF function self-checked; MADLAD inference not run in this session (12 GB model). Not yet run on Power hardware |
