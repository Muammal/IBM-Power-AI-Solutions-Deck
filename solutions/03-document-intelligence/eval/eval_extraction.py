"""eval_extraction.py: field-level accuracy against gold JSON files.

    python eval/eval_extraction.py --docs data/samples --gold data/gold
Gold file name = document stem + .json, containing the correct "data" object (same schema).
Compare the base model and your fine-tuned model by pointing LLM_URL at each server.
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from extract import extract  # noqa: E402

FIELDS = ["vendor_name", "invoice_number", "invoice_date", "currency", "subtotal", "tax", "total"]


def same(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) <= 0.01 * max(1, abs(b))
    return str(a or "").strip().lower() == str(b or "").strip().lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="data/samples")
    ap.add_argument("--gold", default="data/gold")
    a = ap.parse_args()
    correct = {f: 0 for f in FIELDS}
    n = approved = 0
    for g in sorted(pathlib.Path(a.gold).glob("*.json")):
        doc = next(pathlib.Path(a.docs).glob(g.stem + ".*"), None)
        if not doc:
            continue
        gold = json.loads(g.read_text(encoding="utf-8"))
        r = extract(doc)
        n += 1
        approved += r["status"] == "auto_approved"
        for f in FIELDS:
            ok = same(r["data"].get(f), gold.get(f))
            correct[f] += ok
            if not ok:
                print(f"{doc.name}: {f}: got {r['data'].get(f)!r} want {gold.get(f)!r}")
    print(f"\ndocuments: {n}   auto-approved: {approved}")
    for f in FIELDS:
        print(f"  {f:15s} {correct[f] / max(1, n):.2%}")


if __name__ == "__main__":
    main()
