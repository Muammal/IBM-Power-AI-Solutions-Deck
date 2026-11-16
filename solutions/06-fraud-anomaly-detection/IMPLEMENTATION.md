# Solution 06 — Step-by-step implementation

Target: RHEL 9 ppc64le on Power10/Power11, **IBM Cloud PowerVS** or **on-premises** (recommended: next to core banking).
Each step ends with a **Check**.

## Step 1 — Platform

[`common/README.md`](../../common/README.md) C1–C3 with a `small` LPAR (4 cores / 32 GB).
C4–C5 only if you want Granite explanations (`/explain`); a shared Granite server on another LPAR is fine
(set `LLM_URL`).

**Check:** `common/scripts/check_power.sh` → `RESULT: PASS`.

## Step 2 — Python environment

```bash
cd /opt/ibm-power-ai-solutions/solutions/06-fraud-anomaly-detection
../../common/scripts/make_venv.sh && source venv/bin/activate
python -c "import xgboost, sklearn; print(xgboost.__version__, sklearn.__version__)"
```

**Check:** `3.0.2 1.7.0`.

## Step 3 — Train on the synthetic sample

```bash
python data/make_transactions.py
python src/train.py
```

**Check:** metrics JSON with `pr_auc` well above the fraud rate (~0.006) and `recall_at_0.5%_alerts`;
`out/model/` contains `xgb.json`, `iforest.joblib`, `meta.json`.

## Step 4 — Connect your data

Export transactions with these columns (one row per transaction, oldest first is not required):

| column | example | notes |
|---|---|---|
| `txn_id` | 918273 | unique |
| `timestamp` | 2026-10-27 02:14:00 | local time |
| `customer_id` | 42 | pseudonymised ID is enough: no names, no card numbers |
| `channel` | `pos` / `atm` / `mobile_wallet` / `online` / `branch` | map your codes |
| `amount_iqd` | 4500000 | convert USD to IQD |
| `governorate`, `home_governorate` | Basra, Baghdad | |
| `is_foreign`, `new_device` | 0/1 | |
| `is_fraud` | 0/1 | confirmed fraud cases (chargebacks, investigations) |

Set `data.transactions_csv` in `config.yaml` and re-run `python src/train.py`.

**Check:** `pr_auc` on your data. Compare `recall_at_0.5%_alerts` with your current rule system's catch rate.

## Step 5 — Choose thresholds with the fraud team

* `decision.alert_rate`: how many transactions the team can review per day ÷ daily volume.
* `decision.block_probability`: keep high (0.9+) until you trust the model; blocking annoys good customers.

**Check:** fraud team signs off the two numbers.

## Step 6 — Run the scoring API

```bash
python src/score_api.py &
curl -s localhost:8006/score -H 'Content-Type: application/json' -d '{"txn_id": 1, "timestamp": "2026-10-27T02:14:00",
  "customer_id": 42, "channel": "atm", "amount_iqd": 4500000, "governorate": "Basra",
  "home_governorate": "Baghdad", "is_foreign": 0, "new_device": 1}' | jq
curl -s localhost:8006/explain -H 'Content-Type: application/json' -d '<same body>' | jq -r .explanation
```

**Check:** `decision` is `review` or `block` with `top_reasons` such as `amount_vs_cust_median`, `away_from_home`,
`new_device`; `/explain` returns Arabic and English bullets.

## Step 7 — Shadow mode (2–4 weeks)

Score live traffic but do not act: log decisions next to the existing system's alerts. Analysts label the
model's `review` cases in `data/feedback.csv` (`txn_id,label,analyst,decided_at`).

**Check:** the model finds confirmed fraud the current system missed, at an alert volume the team can handle.

## Step 8 — Weekly retraining with analyst feedback

```bash
python finetune/retrain_with_feedback.py
sudo systemctl restart ai-app@06-fraud-anomaly-detection
```

The candidate is promoted only if its PR-AUC is not worse; the previous model is kept in `out/previous/`
for one-command rollback. Schedule it as a weekly systemd timer.

**Check:** `out/model/meta.json` shows the new `trained_at` and metrics.

## Step 9 — Production

```bash
sudo systemctl enable --now ai-app@06-fraud-anomaly-detection
```

* Replace the in-memory `History` in `score_api.py` with your feature store or a database query
  (the comment in the file marks the spot).
* Run several API workers behind your load balancer for high volume (`uvicorn --workers`).
* Keep a model card per version: training dates, metrics, thresholds, approver.

**Check:** `curl -s localhost:8006/health` shows the expected `model_trained_at`.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| PR-AUC near the fraud rate | labels missing or leaked into features | check `is_fraud`; never use post-transaction info as a feature |
| Great test metrics, poor live results | random split instead of time split | keep the time-based split in `train.py` |
| Too many alerts | `alert_rate` too high | lower it; review threshold is recomputed at training time |
| `/explain` timeout | Granite server down/slow | `/score` still works; check `LLM_URL` |
