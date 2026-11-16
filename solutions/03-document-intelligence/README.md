# Solution 03 — Document intelligence: Arabic/English invoices and forms → validated JSON

<div dir="rtl">

**ملخص:** تحويل الفواتير والنماذج الورقية أو الممسوحة ضوئياً (عربي/إنجليزي، أرقام عربية-هندية) إلى بيانات
منظمة (JSON/CSV) مع التحقق التلقائي من المجاميع والتواريخ، وإحالة المستندات المشكوك فيها إلى موظف للمراجعة.
يعمل على المعالج المركزي لخوادم IBM Power داخل الشركة.

</div>

## The problem

Accounts-payable teams in Iraqi companies re-type thousands of supplier invoices, customs papers and
application forms every month. Documents mix Arabic and English, use Arabic-Indic digits (`١٬٠٨٠٬٠٠٠`), arrive as
phone photos or scans, and need checking (do the line items add up?). Errors cost money; the documents are
confidential.

## What it does

```
 scan / photo / PDF ──> src/convert.py ─────────────────────────────> text / Markdown
                        • born-digital PDF: pdftotext                   (Arabic-Indic digits → 0-9)
                        • scan or image: pdftoppm + Tesseract ara+eng
                        • optional Docling: layout, tables, reading order
                                            │
                                            v
                        src/extract.py: Granite 4.2-3B/8B on llama-server
                        JSON-schema-constrained output (schemas/invoice.json)
                                            │
                                            v
                        business rules: qty×price=amount, Σ lines = subtotal,
                        subtotal+tax = total, valid date, known currency
                          │                                  │
                    auto_approved                       needs_review (+ reasons)
                     → ERP / CSV                         → human or review agent
                                                              │ corrected JSON
                                                              v
                                          finetune/build_sft.py → LoRA on Granite 4.2-3B (CPU)
```

## Models and tools

| Role | Model / tool | Licence |
|---|---|---|
| OCR | Tesseract 5, `ara` + `eng` language packs | Apache-2.0 |
| Layout + tables (optional) | Docling (+ optional `ibm-granite/granite-docling-258M` VLM) | MIT / Apache-2.0 |
| Extraction | `ibm-granite/granite-4.2-3b` (Q8_0) or `-8b` (Q4_0) | Apache-2.0 |
| Fine-tuned extractor | LoRA on Granite 4.2-3B with your reviewed documents | yours |

## Compute and compliance

CPU-only. OCR is CPU work by design; extraction uses llama.cpp's MMA kernels. LoRA on the 3B model with
~200 documents is an overnight CPU job on 8 cores. Data stays on-prem. See [`COMPLIANCE.md`](../../COMPLIANCE.md).

| Load | LPAR |
|---|---|
| ≤ 2,000 pages/day | 8 cores, 64 GB (3B extractor on :8081 + 8B shared) |
| ≤ 20,000 pages/day | 16 cores, 128 GB, two 3B llama-server instances, batch at night |

## Quick start (after common steps C1–C5)

```bash
cd /opt/ibm-power-ai-solutions/solutions/03-document-intelligence
../../common/scripts/make_venv.sh && source venv/bin/activate
python src/extract.py data/samples/invoice_001.md            # sample without OCR
python eval/eval_extraction.py --docs data/samples --gold data/gold
python src/batch.py /path/to/scans --out out/
```

## Files

| Path | Purpose |
|---|---|
| `config.yaml` | LLM endpoint, OCR languages, document type, tolerances |
| `schemas/invoice.json` | output schema. Add `schemas/<type>.json` for new document types |
| `src/convert.py` | PDF/image → text (basic or Docling) + digit normalisation |
| `src/extract.py` | LLM extraction + validation rules |
| `src/batch.py` | folder → JSON per file + `summary.csv` + review queue |
| `src/app.py` | FastAPI `POST /extract` |
| `finetune/build_sft.py` | reviewed documents → LoRA training data |
| `eval/eval_extraction.py` | field-level accuracy against gold JSON |
| `requirements-docling.txt` | optional Docling extra (IBM rolling index) |
