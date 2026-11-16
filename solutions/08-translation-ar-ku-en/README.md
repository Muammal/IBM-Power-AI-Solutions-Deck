# Solution 08 — Arabic ↔ Kurdish (Sorani / Kurmanji) ↔ English translation

<div dir="rtl">

**ملخص:** ترجمة آلية بين العربية والكردية (السورانية والكرمانجية) والإنجليزية للوثائق والنماذج والإعلانات
الرسمية، مع قاموس مصطلحات إلزامي، وإمكانية تدريب النموذج على ترجماتكم السابقة. يعمل على المعالج المركزي لخوادم
IBM Power داخل المؤسسة، دون إرسال أي نص إلى خدمة خارجية.

</div>

## The problem

Iraq has two official languages, Arabic and Kurdish, and the Kurdistan Region uses Sorani (Arabic script) and
Kurmanji (Latin script). Federal ministries, the KRG, banks, telecoms and NGOs translate forms, notices, contracts
and customer messages every day. Public cloud translators are not allowed for sensitive text, and most handle
Kurdish poorly.

## What it does

```
 text (ar / ckb / ku / en)
        |
 sentence split ──> MADLAD-400-3B-MT  "<2ckb> ..."  (transformers, bf16 on CPU / MMA)
        |                      ^
        |                      └── optional LoRA adapter trained on YOUR parallel documents
        v
 glossary check (required terms from data/glossary.csv)
        |
        ├─ OK ─────────────> translation
        └─ missing terms ──> optional Granite 4.2 post-edit (Arabic/English targets) ──> translation
```

## Models

| Role | Model | Licence | Notes |
|---|---|---|---|
| Translator | `google/madlad400-3b-mt` (T5, 2.9B) | Apache-2.0 | 400+ languages including `ar`, `ckb`, `ku` |
| Domain adapter | LoRA on MADLAD with your translation memory | yours | `finetune/lora_madlad.py` |
| Post-editor (optional) | `ibm-granite/granite-4.2-8b` | Apache-2.0 | Arabic and English only (Kurdish is not a Granite tested language) |

Excluded: NLLB-200 (good Kurdish, but **CC-BY-NC, non-commercial**).

## Compute and compliance

CPU-only. MADLAD-3B in bf16 needs ~6 GB of RAM. LoRA fine-tuning is CPU-feasible for a few thousand sentence
pairs (hours); larger corpora can use `--gpu` on a compliant machine. See [`COMPLIANCE.md`](../../COMPLIANCE.md).

| Use | LPAR |
|---|---|
| Interactive (a few users) | 8 cores, 64 GB |
| Batch (thousands of pages/day) | 16 cores, 128 GB, run overnight |

## Quick start (after common steps C1–C3)

```bash
cd /opt/ibm-power-ai-solutions/solutions/08-translation-ar-ku-en
../../common/scripts/make_venv.sh && source venv/bin/activate
python src/translate.py --to ckb "يرجى تقديم استمارة الطلب قبل نهاية الشهر"
python src/translate.py --to en  "تم استلام طلبك وهو قيد المراجعة"
python eval/eval_chrf.py data/parallel_sample.tsv
```

## Files

| Path | Purpose |
|---|---|
| `config.yaml` | model, adapter, beams, languages, glossary, post-edit |
| `src/translate.py` | translation, glossary check, post-edit (CLI) |
| `src/app.py` | FastAPI `/translate` |
| `finetune/lora_madlad.py` | LoRA on your parallel TSV (CPU) |
| `eval/eval_chrf.py` | chrF score per language pair |
| `data/glossary.csv` | required terminology (en, ar, ckb, ku) |
| `data/parallel_sample.tsv` | format example for training/eval data |
