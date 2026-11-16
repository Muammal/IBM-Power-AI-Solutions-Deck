# Solution 07 — Customer ticket classification, sentiment and routing (Arabic / Iraqi / English)

<div dir="rtl">

**ملخص:** تصنيف تلقائي لشكاوى وطلبات العملاء (رسائل التطبيق، البريد، وسائل التواصل، نصوص المكالمات) حسب
الموضوع والمشاعر، وتوجيهها إلى الفريق المختص خلال أجزاء من الثانية. يبدأ النموذج الكبير Granite بوضع التصنيفات
الأولية، ثم يُدرَّب نموذج صغير وسريع عليها يعمل على المعالج المركزي لخوادم IBM Power.

</div>

## The problem

Telecom operators, banks and utilities in Iraq receive tens of thousands of messages a day in Modern Standard
Arabic, Iraqi dialect (`ماكو`, `شكد`, `شلون`), English and Arabizi. Manual triage is slow and inconsistent; angry
customers wait in the wrong queue.

## What it does

```
 Phase 1 — bootstrap (no labels yet)
   unlabeled tickets ──> Granite 4.2 (llama-server, JSON enum) ──> llm_labeled.csv ──> human spot-check

 Phase 2 — train the fast model (CPU)
   human labels + LLM labels ──> fine-tune granite-embedding-278m-multilingual (2 heads: category, sentiment)
                              ──> export ONNX (+ int8) ──> parity check

 Phase 3 — serve (ONNX Runtime, MLAS POWER10 kernels, no torch)
   ticket ──> /classify ──> {category, confidence, sentiment, queue, needs_human}
                 low confidence ──> human queue ──> new labels ──> retrain
```

## Models

| Role | Model | Licence | Runtime |
|---|---|---|---|
| Teacher (bootstrap labels) | `ibm-granite/granite-4.2-8b` or `-3b` | Apache-2.0 | llama-server |
| Student classifier | `ibm-granite/granite-embedding-278m-multilingual` + 2 linear heads | Apache-2.0 | torch (training) → ONNX Runtime |

Why distil instead of using the LLM directly? The LLM takes ~1–3 s per ticket on CPU; the 278M classifier takes
milliseconds and needs far less memory, so one LPAR handles the whole contact center.

## Compute and compliance

CPU-only for labelling, training and serving. Fine-tuning a 278M encoder on a few thousand tickets is a
CPU job (tens of minutes). See [`COMPLIANCE.md`](../../COMPLIANCE.md).

| Load | LPAR |
|---|---|
| ≤ 100k tickets/day | 4 cores, 32 GB |
| More | 8 cores, 64 GB, `uvicorn --workers 4`, int8 model |

## Quick start (after common steps C1–C3)

```bash
cd /opt/ibm-power-ai-solutions/solutions/07-ticket-classification
../../common/scripts/make_venv.sh && source venv/bin/activate
python finetune/train_classifier.py          # sample data: 40 tickets (a demo, not a real model)
python src/export_onnx.py --int8
python src/serve.py
```

## Files

| Path | Purpose |
|---|---|
| `config.yaml` | label sets, routing table, model, training settings |
| `data/tickets_sample.csv` | 40 example tickets (MSA, Iraqi dialect, English) |
| `src/llm_label.py` | Granite zero-shot labelling (bootstrap) |
| `src/model.py` | encoder + two heads |
| `finetune/train_classifier.py` | CPU fine-tuning + test report |
| `src/export_onnx.py` | ONNX export, int8 quantisation, parity check |
| `src/serve.py` | FastAPI `/classify`, `/classify_batch` on ONNX Runtime |
