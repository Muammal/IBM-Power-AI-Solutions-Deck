"""llm_label.py: bootstrap training labels with Granite 4.2 (zero-shot, JSON-enum constrained).

You rarely start with thousands of labelled tickets. The LLM labels them (slow, ~1-3 s each on CPU), a person
spot-checks a sample, then a small encoder is trained on those labels (fast, ~10 ms each). This is
"distillation": the big model teaches the small one.

    python src/llm_label.py --limit 2000          # data/unlabeled.csv -> data/llm_labeled.csv
"""
import argparse
import csv
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1] / "common" / "python"))
from llm_client import LLM  # noqa: E402
from power_runtime import load_yaml  # noqa: E402

CFG = load_yaml(ROOT / "config.yaml")


def schema():
    return {"type": "object", "properties": {
        "category": {"type": "string", "enum": CFG["labels"]["category"]},
        "sentiment": {"type": "string", "enum": CFG["labels"]["sentiment"]}},
        "required": ["category", "sentiment"]}


def label(llm, text):
    prompt = ("Classify this customer-service ticket from an Iraqi telecom/bank customer (Arabic, Iraqi dialect or "
              f"English).\nCategories: {', '.join(CFG['labels']['category'])}.\n"
              "complaint = complaints about staff or service quality; network_outage = no service in an area.\n"
              f"Ticket: {text}")
    return llm.chat([{"role": "user", "content": prompt}], json_schema=schema(), temperature=0.0, max_tokens=50)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    llm = LLM(CFG["llm"]["base_url"])
    with open(ROOT / CFG["data"]["unlabeled_csv"], encoding="utf-8") as f:
        texts = [r["text"] for r in csv.DictReader(f)]
    if a.limit:
        texts = texts[: a.limit]
    with open(ROOT / CFG["data"]["llm_labeled_csv"], "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["text", "category", "sentiment"])
        w.writeheader()
        for i, t in enumerate(texts, 1):
            w.writerow({"text": t, **label(llm, t)})
            if i % 50 == 0:
                print(f"{i}/{len(texts)}")
    print("wrote", CFG["data"]["llm_labeled_csv"], "- spot-check 100 rows before training!")


if __name__ == "__main__":
    main()
