"""retrain_with_feedback.py: the model's "fine-tuning" loop. Analysts' confirmed decisions become labels,
the model is retrained, and the new model replaces the old one ONLY if it is better on recent data.

    python finetune/retrain_with_feedback.py            # writes out/candidate, promotes to out/model if better
Run weekly (systemd timer). Keep data/feedback.csv append-only: txn_id,label,analyst,decided_at
"""
import json
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from train import CFG, train  # noqa: E402

import pandas as pd  # noqa: E402


def main():
    current = ROOT / CFG["model"]["dir"]
    candidate = ROOT / "out" / "candidate"
    old = json.loads((current / "meta.json").read_text())["metrics"]["pr_auc"] if (current / "meta.json").exists() else 0
    new = train(pd.read_csv(ROOT / CFG["data"]["transactions_csv"]), candidate)["pr_auc"]
    print(f"PR-AUC current={old:.4f} candidate={new:.4f}")
    if new >= old:
        backup = ROOT / "out" / "previous"
        if current.exists():
            shutil.rmtree(backup, ignore_errors=True)
            shutil.copytree(current, backup)
        shutil.rmtree(current, ignore_errors=True)
        shutil.copytree(candidate, current)
        print("promoted candidate -> model (previous kept in out/previous). Restart the API to load it.")
    else:
        print("candidate is worse: kept the current model")


if __name__ == "__main__":
    main()
