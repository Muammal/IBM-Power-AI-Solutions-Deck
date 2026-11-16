# Solution 05 — Step-by-step implementation

Target: RHEL 9 ppc64le on Power10/Power11, **IBM Cloud PowerVS** or **on-premises**. Each step ends with a **Check**.

## Step 1 — Platform

[`common/README.md`](../../common/README.md) C1–C3. A `small` LPAR (4 cores / 32 GB) is plenty.
No llama.cpp is needed for this solution.

**Check:** `common/scripts/check_power.sh` → `RESULT: PASS`.

## Step 2 — Python environment

```bash
cd /opt/ibm-power-ai-solutions/solutions/05-time-series-forecasting
../../common/scripts/make_venv.sh && source venv/bin/activate
python -c "import tsfm_public, torch; print(torch.__version__)"
```

**Check:** `2.10.0+ppc64le1`.

## Step 3 — Run on the sample data

```bash
python data/make_sample.py
python src/forecast.py
```

The first run downloads the TTM-R2 weights (a few MB) from Hugging Face. Air-gapped: run
`hf download ibm-granite/granite-timeseries-ttm-r2` on a connected machine and copy `~/.cache/huggingface`.

**Check:** JSON with `test_mae` (about 3.6 on the sample) and `out/forecast_next.csv` with 96 rows.

## Step 4 — Prepare your data

Export hourly history (at least 6 months; 2 years captures two summers) as CSV:

```
timestamp,load_mw,temp_c
2024-11-01 00:00:00,58.4,17.2
```

* One row per hour, no gaps (fill short gaps by interpolation, mark long outages).
* Load shedding ("القطع المبرمج") hides real demand: add a column `shed` (0/1) and list it in `control_columns`,
  or train only on unshed hours.
* Several feeders/substations: add `substation_id`, set `id_columns: [substation_id]`: one model learns all.

Edit `config.yaml: data`.

**Check:** `python src/forecast.py` runs on your CSV and prints a zero-shot `test_mae`. This is the baseline.

## Step 5 — Fine-tune with temperature (CPU)

```bash
python finetune/finetune_ttm.py
python src/forecast.py --finetuned
```

Defaults: 20 epochs with early stopping, frozen backbone, exogenous channel mixing for `control_columns`.

**Check:** fine-tuned `test_mae` < zero-shot `test_mae` (on the sample: 2.94 vs 3.60 after 3 epochs).

## Step 6 — Daily forecasting job

Run nightly after the day's data lands:

```bash
python src/forecast.py            # writes out/forecast_next.csv (history-only zero-shot)
```

For the temperature-aware model you need the **weather forecast** for the next 96 h appended to the CSV
(timestamps in the future with `temp_c` filled and `load_mw` empty). Wire this to your weather provider and run
the fine-tuned model in the same job.

**Check:** a fresh `forecast_next.csv` every morning; compare yesterday's forecast with actuals (MAE) in your dashboard.

## Step 7 — API

```bash
sudo systemctl enable --now ai-app@05-time-series-forecasting
curl -s localhost:8005/health
```

`POST /forecast` with `timestamps` and `values` (≥ 512 points) returns the next 96 points.

**Check:** `/health` returns the model and lengths.

## Step 8 — Retraining

Re-run Step 5 monthly, and right after the start of summer. Keep the previous `out/ttm-finetuned` as a backup;
promote the new one only if `test_mae` improves.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ValueError` about length | fewer rows than context + horizon | at least 512 + 96 rows per series |
| Flat forecasts | many gaps / outages | clean gaps, model load shedding as a control column |
| Fine-tuned worse than zero-shot | too little data or too many epochs | more history; early stopping is on; try `use_exogenous: false` |
| Daily vs hourly data | context too long | for daily data use `context_length: 90`, `prediction_length: 30` and set `freq: d` |
