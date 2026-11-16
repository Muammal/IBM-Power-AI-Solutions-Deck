# Solution 07 — Step-by-step implementation

Target: RHEL 9 ppc64le on Power10/Power11, **IBM Cloud PowerVS** or **on-premises**. Each step ends with a **Check**.

## Step 1 — Platform

[`common/README.md`](../../common/README.md) C1–C3 (`small`: 4 cores / 32 GB). For the bootstrap labelling
(Step 4) you also need C4–C5 (a Granite llama-server, can be on another LPAR via `LLM_URL`).

**Check:** `common/scripts/check_power.sh` → `RESULT: PASS`.

## Step 2 — Python environment and base model

```bash
cd /opt/ibm-power-ai-solutions/solutions/07-ticket-classification
../../common/scripts/make_venv.sh && source venv/bin/activate
hf download ibm-granite/granite-embedding-278m-multilingual
python -c "import onnxruntime as o; print(o.__version__, o.get_available_providers())"
```

**Check:** `1.23.1 ['CPUExecutionProvider']`.

## Step 3 — Pipeline smoke test on the sample

```bash
python finetune/train_classifier.py
python src/export_onnx.py --int8
python src/serve.py &
curl -s localhost:8007/classify -H 'Content-Type: application/json' -d '{"text":"ماكو انترنت بالكرادة من الصبح"}' | jq
```

40 tickets cannot make a good model; this step only proves the pipeline works end to end.

**Check:** `max |torch - onnx|` below `1e-3`; the API returns JSON with `category`, `queue`, `sentiment`.

## Step 4 — Agree the label set

Edit `labels` and `routing` in `config.yaml` with the contact-center managers. Keep 5–12 categories with
clear definitions (write one sentence each; put the definitions into `src/llm_label.py`'s prompt).

**Check:** managers sign off categories + definitions.

## Step 5 — Bootstrap labels with Granite

Export 2,000–10,000 recent tickets to `data/unlabeled.csv` (one column `text`; remove phone numbers and names).

```bash
python src/llm_label.py --limit 3000
```

Then have two agents label the same 200 tickets by hand into `data/tickets_sample.csv` (replace the demo rows)
and measure how often the LLM agrees with them.

**Check:** LLM-vs-human agreement on category ≥ 80%. If lower, improve the category definitions and re-run.

## Step 6 — Train the classifier

```bash
python finetune/train_classifier.py --with-llm-labels
```

The test set is always human-labelled tickets only. Read the per-category precision/recall.

**Check:** macro-F1 on category meets your target (e.g. ≥ 0.85); no category below 0.6 recall.

## Step 7 — Export and benchmark

```bash
python src/export_onnx.py --int8
ONNX_MODEL=out/classifier/model.int8.onnx python src/serve.py &
time (for i in $(seq 100); do curl -s localhost:8007/classify -H 'Content-Type: application/json' \
  -d '{"text":"الفاتورة عالية هالشهر"}' >/dev/null; done)
```

Use int8 only if its test accuracy is within 1 point of fp32 (re-run Step 6's report against it if needed).

**Check:** latency per ticket in milliseconds; int8 accuracy acceptable.

## Step 8 — Production and continuous learning

```bash
sudo systemctl enable --now ai-app@07-ticket-classification
```

* Your ticketing system calls `/classify` on every new ticket and routes by `queue`.
* `needs_human: true` (confidence < 0.6) goes to a triage person; their label is appended to the labelled CSV.
* Retrain monthly (Step 6–7). GPU is never needed for this model size.

**Check:** after a week, compare auto-routing accuracy with the triage team's corrections.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| One category swallows others | class imbalance | add examples of small classes; merge rare ones into `other` |
| Dialect tickets misclassified | few dialect examples | add Iraqi-dialect examples to the human set |
| ONNX parity error > 1e-3 | dropout left on / wrong opset | the script calls `model.eval()`; keep opset 17 |
| Slow training | too many threads / SMT8 | SMT4, threads = physical cores |
