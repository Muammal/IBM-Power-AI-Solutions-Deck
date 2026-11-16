# Solution 03 — Step-by-step implementation

Target: RHEL 9 ppc64le, IBM Power10/Power11, on **IBM Cloud PowerVS** or **on-premises**. Each step ends with a **Check**.

## Step 1 — Platform

Follow [`common/README.md`](../../common/README.md) C1–C5 (`medium` LPAR: 8 cores / 64 GB).
`os_prep.sh` installs **poppler-utils** and **Tesseract with Arabic** (from EPEL on RHEL 9).

**Check:**

```bash
tesseract --list-langs | grep -E '^(ara|eng)$'      # both listed
pdftotext -v 2>&1 | head -1
curl -s localhost:8080/health
```

If EPEL is not allowed in your organisation, download `tesseract`, `leptonica` and `tesseract-langpack-ara`
RPMs on an approved mirror, or build Tesseract from source.

## Step 2 — A dedicated 3B extractor (recommended)

Extraction is short, structured output: the 3B model is accurate enough and much faster.

```bash
common/scripts/get_model.sh ibm-granite/granite-4.2-3b-GGUF granite-4.2-3b-Q8_0.gguf
PARALLEL=4 common/scripts/run_llama_server.sh /opt/models/granite-4.2-3b-Q8_0.gguf 8081 16384 &
echo 'LLM_URL=http://127.0.0.1:8081/v1' > solutions/03-document-intelligence/.env
export LLM_URL=http://127.0.0.1:8081/v1
```

**Check:** `curl -s localhost:8081/health` → ok.

## Step 3 — Python environment

```bash
cd /opt/ibm-power-ai-solutions/solutions/03-document-intelligence
../../common/scripts/make_venv.sh && source venv/bin/activate
```

Optional Docling (better tables and reading order, uses IBM's **rolling** wheel index):

```bash
../../common/scripts/make_venv.sh requirements-docling.txt venv-docling
```

**Check:** `python -c "import fastapi, httpx, yaml; print('ok')"`.

## Step 4 — Extract the sample

```bash
python src/extract.py data/samples/invoice_001.md
```

**Check:** `"status": "auto_approved"`, `invoice_date` = `2026-10-12`, `total` = `1080000`, `currency` = `IQD`.

## Step 5 — OCR a real scan

```bash
python src/convert.py /path/to/scan.pdf | head -40
python src/extract.py /path/to/scan.pdf
```

Tips for Iraqi documents: scan at 300 dpi, grey-scale; phone photos should be flat and well lit.
For Kurdish (Sorani) forms install the `ckb` traineddata from the tessdata repository and add `ckb` to
`converter.ocr_languages`.

**Check:** the Arabic text in `convert.py` output is readable (not reversed or broken into single letters).

## Step 6 — Measure accuracy

Create 30–50 gold files: run `extract.py`, correct the JSON by hand, save the `data` object to
`data/gold/<document-stem>.json` and the document in `data/samples/`.

```bash
python eval/eval_extraction.py --docs data/samples --gold data/gold
```

**Check:** per-field accuracy printed. This is your baseline.

## Step 7 — Batch processing and review queue

```bash
python src/batch.py /data/inbox --out /data/extracted
```

* `auto_approved` JSON files go to `/data/extracted/`; `needs_review` files go to `/data/extracted/review/`.
* `summary.csv` opens in Excel with Arabic intact (UTF-8 with BOM).
* Reviewers correct the `data` object and move the file to `data/reviewed/` (this feeds Step 8).

**Check:** `summary.csv` has one row per document.

## Step 8 — Fine-tune the extractor on your reviewed documents (CPU)

After 200+ reviewed documents:

```bash
python finetune/build_sft.py
../../common/scripts/make_venv.sh ../../common/requirements-finetune.txt /opt/venv-finetune
source /opt/venv-finetune/bin/activate
python ../../common/python/lora_sft.py --data data/sft_train.jsonl --eval-data data/sft_eval.jsonl \
    --model ibm-granite/granite-4.2-3b --out out/lora-invoice --epochs 3 --merge
../../common/scripts/hf_to_gguf.sh out/lora-invoice/merged /opt/models/granite-4.2-3b-invoice Q8_0
PARALLEL=4 ../../common/scripts/run_llama_server.sh /opt/models/granite-4.2-3b-invoice-Q8_0.gguf 8082 &
LLM_URL=http://127.0.0.1:8082/v1 python eval/eval_extraction.py
```

Switch `LLM_URL` to the fine-tuned server only if it beats the Step 6 baseline.
GPU option: `--gpu` on a separate compliant machine (see [`COMPLIANCE.md`](../../COMPLIANCE.md)).

**Check:** field accuracy ≥ baseline and auto-approved share goes up.

## Step 9 — Production

```bash
sudo systemctl enable --now ai-app@03-document-intelligence
curl -s -F file=@data/samples/invoice_001.md localhost:8003/extract | jq .status
```

* Connect `auto_approved` output to your ERP import (CSV or API). Keep `needs_review` human-approved.
* Keep original scans read-only for audit; store the JSON next to them.

**Check:** the curl prints `"auto_approved"`.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `tesseract not found` | EPEL missing | Step 1 note |
| Arabic text appears reversed | viewing in a terminal without bidi | open the output in an editor; the model reads logical order correctly |
| Totals off by ×1000 | `٬` / `,` thousands separators | `normalize_digits` handles `٬`; check unusual separators in your scans |
| Many `needs_review` for one vendor | unusual layout | add 10–20 reviewed examples of that vendor and re-run Step 8 |
| Slow (> 20 s per page) | 8B + thinking | use the 3B server (Step 2); thinking is off by default |
