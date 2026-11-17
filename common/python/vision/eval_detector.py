"""eval_detector.py: per-class AP50 and precision/recall at your alert thresholds, on a held-out COCO set.

    python common/python/vision/eval_detector.py --model models/detector/model.onnx --data data/test \
        --thresholds '{"handgun": 0.6, "rifle": 0.6, "knife": 0.7}' --gate '{"handgun": {"recall": 0.85, "precision": 0.9}}'
Exit code 1 when a gate fails: use it as the release check before a new model goes live.
The test set must come from YOUR cameras (angles, lighting, IR at night) and never be used for training.
"""
import argparse
import json
import pathlib
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from vision.detector import OnnxDetector, iou  # noqa: E402


def ap50(dets, gts, n_gt):
    """dets: [(score, image_id, box)], gts: {image_id: [box, ...]} -> all-point interpolated AP at IoU 0.5."""
    used = {k: [False] * len(v) for k, v in gts.items()}
    tp = []
    for score, img, box in sorted(dets, key=lambda d: -d[0]):
        cands = gts.get(img, [])
        best, bi = 0.0, -1
        for j, g in enumerate(cands):
            v = iou(box, g)
            if v > best and not used[img][j]:
                best, bi = v, j
        if best >= 0.5:
            used[img][bi] = True
            tp.append(1)
        else:
            tp.append(0)
    if not n_gt:
        return 0.0
    tp = np.array(tp, float)
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    rec, prec = ctp / n_gt, ctp / np.maximum(ctp + cfp, 1e-9)
    mrec, mpre = np.concatenate([[0], rec, [1]]), np.concatenate([[1], prec, [0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--thresholds", default="{}")
    ap.add_argument("--gate", default="{}")
    ap.add_argument("--batch", type=int, default=4)
    a = ap.parse_args()
    thresholds, gate = json.loads(a.thresholds), json.loads(a.gate)
    folder = pathlib.Path(a.data)
    coco = json.loads((folder / "annotations.json").read_text(encoding="utf-8"))
    cat_name = {c["id"]: c["name"] for c in coco["categories"]}
    gts = {}
    for an in coco["annotations"]:
        x, y, w, h = an["bbox"]
        gts.setdefault(cat_name[an["category_id"]], {}).setdefault(an["image_id"], []).append((x, y, x + w, y + h))
    det = OnnxDetector(a.model, min_score=0.05)
    found = {}
    imgs = coco["images"]
    for i in range(0, len(imgs), a.batch):
        chunk = imgs[i:i + a.batch]
        frames = [np.asarray(Image.open(folder / "images" / pathlib.Path(m["file_name"]).name).convert("RGB")) for m in chunk]
        for m, ds in zip(chunk, det.detect(frames)):
            for d in ds:
                found.setdefault(d.label, []).append((d.score, m["id"], d.box))
    report, failed = {}, []
    for label in sorted(set(gts) | set(thresholds)):
        g = gts.get(label, {})
        n_gt = sum(len(v) for v in g.values())
        dets = found.get(label, [])
        thr = thresholds.get(label, 0.5)
        kept = [d for d in dets if d[0] >= thr]
        hits = 0
        for img, boxes in g.items():
            for b in boxes:
                hits += any(d[1] == img and iou(d[2], b) >= 0.5 for d in kept)
        tp_det = sum(any(iou(d[2], b) >= 0.5 for b in g.get(d[1], [])) for d in kept)
        r = {"gt": n_gt, "ap50": round(ap50(dets, g, n_gt), 4), "threshold": thr,
             "recall": round(hits / max(1, n_gt), 4), "precision": round(tp_det / max(1, len(kept)), 4)}
        report[label] = r
        for k, v in gate.get(label, {}).items():
            if r[k] < v:
                failed.append(f"{label}.{k}={r[k]} < {v}")
    print(json.dumps(report, indent=2))
    if failed:
        print("GATE FAILED:", "; ".join(failed))
        sys.exit(1)
    print("GATE PASSED" if gate else "(no gate given)")


if __name__ == "__main__":
    main()
