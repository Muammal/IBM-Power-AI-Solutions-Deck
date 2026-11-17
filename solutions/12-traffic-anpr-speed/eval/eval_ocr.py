"""eval_ocr.py: plate-reading accuracy on a held-out labelled crop set (release gate for a new OCR model).

    python eval/eval_ocr.py data/ocr/test --min-exact 0.97
Exact match after normalisation is the only metric that matters for enforcement: one wrong character = wrong car.
Also reports accuracy among reads above ocr.min_confidence (what the system actually uses for violations).
"""
import argparse
import csv
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parents[1] / "common" / "python"))
import cv2  # noqa: E402
from power_runtime import load_yaml  # noqa: E402

from ocr import OnnxPlateReader  # noqa: E402
from plates import normalise, repair  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--min-exact", type=float)
    ap.add_argument("--min-precision-confident", type=float, default=0.995)
    a = ap.parse_args()
    cfg = load_yaml(ROOT / "config.yaml")
    reader = OnnxPlateReader(ROOT / cfg["ocr"]["model"])
    folder = pathlib.Path(a.folder)
    with open(folder / "labels.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    exact = conf_n = conf_ok = 0
    for r in rows:
        raw, conf = reader.read(cv2.cvtColor(cv2.imread(str(folder / r["file"])), cv2.COLOR_BGR2RGB))
        got = repair(normalise(raw), cfg["ocr"]["formats"])
        ok = got == normalise(r["text"])
        exact += ok
        if conf >= cfg["ocr"]["min_confidence"]:
            conf_n += 1
            conf_ok += ok
        elif not ok:
            pass
    n = max(1, len(rows))
    prec = conf_ok / max(1, conf_n)
    print(f"crops={len(rows)} exact={exact / n:.4f} confident_share={conf_n / n:.3f} precision_when_confident={prec:.4f}")
    fail = (a.min_exact is not None and exact / n < a.min_exact) or prec < a.min_precision_confident
    print("GATE FAILED" if fail else "GATE PASSED")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
