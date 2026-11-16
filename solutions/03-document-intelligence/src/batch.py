"""batch.py: process a whole folder. Writes out/<file>.json per document and out/summary.csv.

    python src/batch.py inbox/ --out out/
Documents with status needs_review go to the review queue (out/review/) for a person or the review agent.
"""
import argparse
import csv
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from extract import extract  # noqa: E402

SUPPORTED = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".md", ".txt"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inbox")
    ap.add_argument("--out", default="out")
    ap.add_argument("--type")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    (out / "review").mkdir(parents=True, exist_ok=True)
    rows = []
    for f in sorted(pathlib.Path(a.inbox).rglob("*")):
        if f.suffix.lower() not in SUPPORTED:
            continue
        try:
            r = extract(f, a.type)
        except Exception as exc:  # one broken scan must not stop the batch
            rows.append({"file": str(f), "status": "error", "problems": str(exc)})
            continue
        target = out / ("review" if r["status"] == "needs_review" else ".") / f"{f.stem}.json"
        target.write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
        rows.append({"file": str(f), "status": r["status"], "problems": "; ".join(r["problems"]),
                     "total": r["data"].get("total"), "currency": r["data"].get("currency")})
        print(f"{r['status']:14s} {f.name}")
    with open(out / "summary.csv", "w", newline="", encoding="utf-8-sig") as fh:  # utf-8-sig: Excel shows Arabic
        w = csv.DictWriter(fh, fieldnames=["file", "status", "problems", "total", "currency"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} documents -> {out / 'summary.csv'}")


if __name__ == "__main__":
    main()
