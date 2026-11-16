"""score_api.py: real-time scoring with reasons, plus an optional Granite 4.2 explanation for analysts.

    python src/score_api.py
    curl -s localhost:8006/score -H 'Content-Type: application/json' -d '{"txn_id": 1, "timestamp": "2026-10-27T02:14:00",
      "customer_id": 42, "channel": "atm", "amount_iqd": 4500000, "governorate": "Basra",
      "home_governorate": "Baghdad", "is_foreign": 0, "new_device": 1}' | jq

Customer history for behavioural features is kept in memory (last 30 days from the CSV + scored traffic).
In production replace `History` with your feature store or a database query.
"""
import collections
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1] / "common" / "python"))
sys.path.insert(0, str(ROOT / "src"))
from power_runtime import configure_threads, load_yaml  # noqa: E402

CFG = load_yaml(ROOT / "config.yaml")
configure_threads(CFG["runtime"]["threads"])

import joblib  # noqa: E402
import pandas as pd  # noqa: E402
import uvicorn  # noqa: E402
import xgboost as xgb  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from features import build_features  # noqa: E402
from llm_client import LLM  # noqa: E402

MODEL_DIR = ROOT / CFG["model"]["dir"]
META = json.loads((MODEL_DIR / "meta.json").read_text(encoding="utf-8"))
BOOSTER = xgb.Booster()
BOOSTER.load_model(MODEL_DIR / "xgb.json")
ISO = joblib.load(MODEL_DIR / "iforest.joblib") if (MODEL_DIR / "iforest.joblib").exists() else None


class Txn(BaseModel):
    txn_id: int
    timestamp: str
    customer_id: int
    channel: str = Field(pattern="^(pos|atm|mobile_wallet|online|branch)$")
    amount_iqd: float = Field(gt=0)
    governorate: str
    home_governorate: str
    is_foreign: int = 0
    new_device: int = 0


class History:
    def __init__(self, csv, days=30, max_per_customer=200):
        self.rows = collections.defaultdict(lambda: collections.deque(maxlen=max_per_customer))
        if csv.exists():
            df = pd.read_csv(csv, parse_dates=["timestamp"])
            df = df[df["timestamp"] >= df["timestamp"].max() - pd.Timedelta(days=days)]
            for r in df.drop(columns=["is_fraud"], errors="ignore").to_dict("records"):
                self.rows[r["customer_id"]].append(r)

    def features_for(self, txn):
        hist = list(self.rows[txn["customer_id"]]) + [txn]
        X = build_features(pd.DataFrame(hist)).loc[[len(hist) - 1]]  # the new transaction's row
        self.rows[txn["customer_id"]].append(txn)
        return X


HISTORY = History(ROOT / CFG["data"]["transactions_csv"])
app = FastAPI(title="Fraud scoring on IBM Power")


def score(txn):
    X = HISTORY.features_for(dict(txn, timestamp=pd.Timestamp(txn["timestamp"])))
    if ISO is not None:
        X = X.assign(iso_score=-ISO.score_samples(X))
    d = xgb.DMatrix(X[META["features"]])
    p = float(BOOSTER.predict(d)[0])
    contrib = BOOSTER.predict(d, pred_contribs=True)[0][:-1]  # SHAP values per feature (last = bias)
    reasons = sorted(zip(META["features"], contrib, X[META["features"]].iloc[0]), key=lambda t: -t[1])[:4]
    decision = "block" if p >= META["block_threshold"] else "review" if p >= META["review_threshold"] else "allow"
    return {"txn_id": txn["txn_id"], "fraud_probability": round(p, 4), "decision": decision,
            "top_reasons": [{"feature": f, "value": round(float(v), 3), "impact": round(float(c), 3)}
                            for f, c, v in reasons if c > 0]}


@app.get("/health")
def health():
    return {"status": "ok", "model_trained_at": META["trained_at"], "metrics": META["metrics"]}


@app.post("/score")
def score_endpoint(txn: Txn):
    return score(txn.model_dump())


@app.post("/explain")
def explain(txn: Txn):
    """Score + a short Arabic/English explanation for the fraud analyst (Granite 4.2)."""
    result = score(txn.model_dump())
    if CFG["llm"]["enabled"] and result["decision"] != "allow":
        prompt = ("You help a bank fraud analyst in Iraq. Explain in 3 short bullet points in Arabic, then the same "
                  "in English, why this transaction was flagged. Use only the facts given. Do not accuse the "
                  f"customer; suggest what to verify.\nTransaction: {txn.model_dump_json()}\nModel output: {json.dumps(result)}")
        result["explanation"] = LLM(CFG["llm"]["base_url"]).chat([{"role": "user", "content": prompt}], max_tokens=400)
    return result


if __name__ == "__main__":
    uvicorn.run(app, host=CFG["server"]["host"], port=int(CFG["server"]["port"]))
