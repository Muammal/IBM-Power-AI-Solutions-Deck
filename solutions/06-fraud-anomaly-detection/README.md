# Solution 06 — Real-time fraud and anomaly detection for banks and wallets

<div dir="rtl">

**ملخص:** نظام يقيّم كل عملية دفع (بطاقة، صراف آلي، محفظة إلكترونية، تحويل) خلال أجزاء من الثانية، ويحدد
العمليات المشبوهة مع أسباب واضحة، ويكتب شرحاً بالعربية والإنجليزية لمحلل الاحتيال. يتعلم النموذج من قرارات
المحللين أسبوعياً. يعمل على المعالج المركزي لخوادم IBM Power في مركز بيانات المصرف.

</div>

## The problem

Electronic payments in Iraq (cards, ATMs, mobile wallets, salary cards) have grown fast, and so has fraud:
stolen cards used at night, SIM-swap takeovers of wallets, withdrawals in a governorate the customer never visits.
Rule-only systems flag too much or too little, and fraud analysts drown in alerts. Payment data cannot leave the
bank (Central Bank of Iraq supervision).

## What it does

```
 transaction (channel, amount IQD, governorate, device, time)
        |
 src/features.py   behaviour vs the customer's OWN history: amount / usual amount, txns in last 1h/24h,
        |          away from home governorate, new device, night, foreign online
        v
 Isolation Forest (unsupervised "how unusual") ──> iso_score feature
        |
 XGBoost (supervised, IBM ppc64le wheel) ──> fraud probability + SHAP reasons (pred_contribs)
        |
 decision: allow / review (top 0.5% for analysts) / block (p ≥ 0.90)
        |
 /explain: Granite 4.2-8B writes a 3-bullet explanation in Arabic + English for the analyst
        |
 analyst decision ──> data/feedback.csv ──> finetune/retrain_with_feedback.py (weekly, promote only if better)
```

## Models

| Role | Model | Licence | Runtime |
|---|---|---|---|
| Fraud classifier | XGBoost 3.0.2 (`hist`), trained on your data | Apache-2.0 | IBM `xgboost==3.0.2+ppc64le1` |
| Anomaly score | scikit-learn Isolation Forest | BSD-3 | IBM `scikit-learn==1.7.0+ppc64le1` |
| Analyst explanation | `ibm-granite/granite-4.2-8b` (GGUF Q4_0) | Apache-2.0 | llama-server (MMA) |
| Optional, metric streams | `ibm-granite/granite-timeseries-tspulse-r1` | Apache-2.0 | granite-tsfm (see Solution 05) |

"Fine-tuning" here is the weekly retraining loop on analyst-confirmed labels: the standard and most effective way
to keep a fraud model current.

## Compute and compliance

CPU-only. XGBoost training on 300k transactions takes minutes; scoring takes milliseconds per transaction.
The bank's core-banking LPARs are often already on IBM Power: this runs next to them. See [`COMPLIANCE.md`](../../COMPLIANCE.md).

| Volume | LPAR |
|---|---|
| ≤ 1M transactions/day | 4 cores, 32 GB (+ shared Granite server for `/explain`) |
| ≤ 20M transactions/day | 8–16 cores, 64–128 GB, several API workers |

## Quick start (after common steps C1–C3; C5 for `/explain`)

```bash
cd /opt/ibm-power-ai-solutions/solutions/06-fraud-anomaly-detection
../../common/scripts/make_venv.sh && source venv/bin/activate
python data/make_transactions.py
python src/train.py
python src/score_api.py            # 127.0.0.1:8006  /score /explain /health
```

## Files

| Path | Purpose |
|---|---|
| `config.yaml` | data paths, XGBoost parameters, alert rate, block threshold |
| `data/make_transactions.py` | synthetic IQD transactions with injected fraud patterns |
| `src/features.py` | behavioural features (same code for training and scoring) |
| `src/train.py` | time-based split, Isolation Forest + XGBoost, metrics, thresholds |
| `src/score_api.py` | FastAPI `/score` (reasons) and `/explain` (Granite narrative) |
| `finetune/retrain_with_feedback.py` | analyst labels → retrain → promote if PR-AUC is not worse |
