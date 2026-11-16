# AGENTS.md — Solution 06 (Fraud & anomaly detection)

Three parts: **agent brief**, **agentic workflow**, **change log**. Repository rules: [`/AGENTS.md`](../../AGENTS.md).

---

## 1. Agent brief

**Purpose:** score payment transactions for fraud in real time, with reasons, and learn weekly from analysts.

**Invariants:**

* `src/features.py` is shared by training and scoring. Any feature change must work for both, and must use only
  information available **at transaction time** (no look-ahead, no labels, no post-transaction data).
* The train/test split stays **time-based**.
* A retrained model is promoted only through `finetune/retrain_with_feedback.py` (metric gate + backup).
* The LLM never makes the decision. It only explains a decision already made by the model and thresholds.
* No names, phone numbers or card numbers in features, logs or LLM prompts. `customer_id` is pseudonymous.
* CPU-only, IBM ppc64le wheels for xgboost/sklearn/pandas.

**Map:**

| Change | File |
|---|---|
| New feature | `src/features.py: FEATURES + build_features()` then retrain |
| Model parameters, thresholds | `config.yaml` |
| API / explanation prompt | `src/score_api.py` |
| Retraining policy | `finetune/retrain_with_feedback.py` |

**Verify:**

```bash
python -m py_compile src/*.py finetune/*.py data/*.py
python data/make_transactions.py && python src/train.py      # pr_auc must not drop vs the change-log value
python -c "import sys; sys.path.insert(0,'src'); import score_api as s; print(s.score({'txn_id':1,'timestamp':'2026-10-27T02:14:00','customer_id':42,'channel':'atm','amount_iqd':4500000,'governorate':'Basra','home_governorate':'Baghdad','is_foreign':0,'new_device':1}))"
```

---

## 2. Agentic workflow (runtime)

An **investigation agent** prepares each `review` case for the analyst, so the analyst decides in one minute
instead of ten.

```
 transaction with decision == review
        |
        v
 [Investigation agent] Granite 4.2-8B, thinking=on, max 6 tool calls, read-only tools
   tools:
     customer_recent_activity(customer_id, days)  -> last N transactions (amounts, channels, governorates)
     device_history(customer_id)                  -> devices and first-seen dates
     similar_confirmed_cases(features)            -> nearest past confirmed frauds (pattern match)
     score_counterfactual(txn, changes)           -> model score if e.g. the device were known (sensitivity)
        |
        v
 case file (Arabic + English): facts, pattern match, recommended verification ("call the customer about
 the ATM withdrawal in Basra at 02:14")
        |
        v
 analyst decides: fraud / legitimate  ──> data/feedback.csv  ──> weekly retraining
```

**Guardrails:** read-only tools; the agent cannot block cards, contact customers or change thresholds; every case
file cites the tool outputs it used; personal identifiers are never passed to the model.

**Implementation path:** Solution 09 `src/agent.py` (tool loop + audit log). Tools are thin Python functions over
your transaction store.

---

## 3. Agent change protocol

1. Change, run every **Verify** command, note the new `pr_auc`.
2. Docs changed → `python tools/build_docx.py solutions/06-fraud-anomaly-detection`.
3. Append a row below.

### Agent change log

| Date | Agent / author | Change | Verification result |
|---|---|---|---|
| 2026-11-15 | Claude Code (Opus 5.5) | Initial version: synthetic data, behavioural features, Isolation Forest + XGBoost, scoring API with SHAP reasons, Granite explanations, feedback retraining | Run on x86 CPU with the same library versions: 300k synthetic transactions, PR-AUC 0.749 (fraud rate 0.6%), ROC-AUC 0.991, recall 0.67 at 0.5% alerts, training 3.3 s; `/score` 28–40 ms incl. features; retraining gate kept the better model. Not yet run on Power hardware |
