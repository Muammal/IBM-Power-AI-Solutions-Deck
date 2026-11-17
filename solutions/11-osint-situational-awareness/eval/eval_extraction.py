"""eval_extraction.py: event-extraction quality against analyst-labelled examples (release gate for prompt/model).

    python eval/eval_extraction.py data/events_gold.jsonl --min-category 0.8 --min-governorate-f1 0.8
Grow the gold file to 200+ real items from your own sources (analyst corrections are exactly that).
Metrics: is_event accuracy, category accuracy (events only), governorate set F1, severity within +/-1.
"""
import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from app import CFG  # noqa: E402
from enrich import extract_event, new_llm  # noqa: E402


def f1(pred, gold):
    pred, gold = set(pred), set(gold)
    if not pred and not gold:
        return 1.0
    tp = len(pred & gold)
    return 0.0 if tp == 0 else 2 * tp / (len(pred) + len(gold))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gold")
    ap.add_argument("--min-category", type=float)
    ap.add_argument("--min-governorate-f1", type=float)
    a = ap.parse_args()
    llm = new_llm(CFG)
    rows = [json.loads(line) for line in open(a.gold, encoding="utf-8") if line.strip()]
    ev_ok = cat_ok = sev_ok = n_ev = 0
    gov = []
    for r in rows:
        p = extract_event({"title": r["title"], "text": r["text"], "source": "gold"}, llm)
        ev_ok += p["is_event"] == r["is_event"]
        if r["is_event"]:
            n_ev += 1
            cat_ok += p["category"] == r["category"]
            sev_ok += abs(int(p["severity"]) - int(r["severity"])) <= 1
            gov.append(f1(p["governorates"], r["governorates"]))
            if p["category"] != r["category"]:
                print(f"category: got {p['category']} want {r['category']}: {r['title'][:70]}")
    res = {"items": len(rows), "is_event_acc": ev_ok / len(rows), "category_acc": cat_ok / max(1, n_ev),
           "governorate_f1": sum(gov) / max(1, len(gov)), "severity_within_1": sev_ok / max(1, n_ev)}
    print(json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in res.items()}, indent=2))
    fail = (a.min_category and res["category_acc"] < a.min_category) or \
           (a.min_governorate_f1 and res["governorate_f1"] < a.min_governorate_f1)
    print("GATE FAILED" if fail else "GATE PASSED")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
