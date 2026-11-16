"""train.py: train the fraud model (XGBoost + Isolation Forest anomaly score), time-based evaluation.

    python data/make_transactions.py && python src/train.py
Writes out/model/: xgb.json, iforest.joblib, meta.json (features, thresholds, metrics).
"""
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1] / "common" / "python"))
sys.path.insert(0, str(ROOT / "src"))
from power_runtime import configure_threads, load_yaml  # noqa: E402

CFG = load_yaml(ROOT / "config.yaml")
THREADS = configure_threads(CFG["runtime"]["threads"])

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402
from sklearn.ensemble import IsolationForest  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from features import FEATURES, build_features  # noqa: E402


def load_labels(df):
    """Analyst feedback overrides the original labels (that is how the model learns from investigations)."""
    fb = ROOT / CFG["data"]["feedback_csv"]
    if fb.exists():
        f = pd.read_csv(fb).drop_duplicates("txn_id", keep="last").set_index("txn_id")["label"]
        hit = df["txn_id"].isin(f.index)
        df.loc[hit, "is_fraud"] = df.loc[hit, "txn_id"].map(f).astype(int)
        print(f"applied {hit.sum()} analyst labels from {fb.name}")
    return df


def recall_at_rate(y, p, rate):
    k = max(1, int(len(p) * rate))
    top = np.argsort(-p)[:k]
    return y[top].sum() / max(1, y.sum()), float(np.sort(p)[-k])


def train(df, out_dir):
    df = load_labels(df.sort_values("timestamp").reset_index(drop=True))
    X = build_features(df)
    y = df.loc[X.index, "is_fraud"].to_numpy()
    cut = int(len(X) * (1 - CFG["data"]["test_fraction"]))  # past -> train, most recent -> test
    Xtr, Xte, ytr, yte = X.iloc[:cut], X.iloc[cut:], y[:cut], y[cut:]

    m = CFG["model"]
    iso = None
    if m["isolation_forest"]:
        iso = IsolationForest(n_estimators=200, contamination="auto", random_state=0, n_jobs=THREADS or None)
        iso.fit(Xtr[ytr == 0])  # learn "normal" only
        Xtr = Xtr.assign(iso_score=-iso.score_samples(Xtr))
        Xte = Xte.assign(iso_score=-iso.score_samples(Xte))

    t = time.time()
    val = int(len(Xtr) * 0.9)  # last 10% of the training period picks the number of trees (test stays unseen)
    clf = xgb.XGBClassifier(**m["xgboost"], scale_pos_weight=(ytr == 0).sum() / max(1, ytr.sum()),
                            eval_metric="aucpr", n_jobs=THREADS or None, early_stopping_rounds=50)
    clf.fit(Xtr.iloc[:val], ytr[:val], eval_set=[(Xtr.iloc[val:], ytr[val:])], verbose=100)
    p = clf.predict_proba(Xte)[:, 1]
    rec, review_thr = recall_at_rate(yte, p, CFG["decision"]["alert_rate"])
    metrics = {"pr_auc": float(average_precision_score(yte, p)), "roc_auc": float(roc_auc_score(yte, p)),
               f"recall_at_{CFG['decision']['alert_rate']:.1%}_alerts": float(rec),
               "train_rows": int(len(Xtr)), "test_rows": int(len(Xte)), "train_seconds": round(time.time() - t, 1)}
    print(json.dumps(metrics, indent=2))

    out_dir.mkdir(parents=True, exist_ok=True)
    clf.save_model(out_dir / "xgb.json")
    if iso is not None:
        joblib.dump(iso, out_dir / "iforest.joblib")
    meta = {"features": list(Xtr.columns), "review_threshold": review_thr,
            "block_threshold": CFG["decision"]["block_probability"], "metrics": metrics,
            "trained_at": pd.Timestamp.now().isoformat(timespec="seconds")}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    data = pd.read_csv(ROOT / CFG["data"]["transactions_csv"])
    train(data, ROOT / CFG["model"]["dir"])
    print("saved to", ROOT / CFG["model"]["dir"])
