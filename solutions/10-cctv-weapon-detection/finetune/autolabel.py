"""autolabel.py: use the zero-shot detector (OWLv2) as a TEACHER to pre-label your own recordings.

Labelling thousands of frames by hand is the slowest part of Step 5. This script samples frames from recordings,
runs OWLv2 with the configured text queries, and writes a COCO dataset with the boxes as PRE-labels. People then
correct them in CVAT (much faster than drawing every box), and the result trains the fast RT-DETRv2 student
(common/python/vision/train_rtdetr.py). Pre-labels are never used for training without human review.

    python finetune/autolabel.py /data/recordings/staged --every-s 1 --out data/prelabels/staged
    # CVAT: create a task from data/prelabels/staged/images, import annotations.json (COCO 1.0), review, export
Needs the zero-shot environment (requirements-zeroshot.txt).
"""
import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import av  # noqa: E402
import cv2  # noqa: E402

from app import CFG  # noqa: E402
from vision.zeroshot import ZeroShotDetector  # noqa: E402

VIDEO = {".mp4", ".mkv", ".avi", ".mov", ".ts", ".webm"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recordings")
    ap.add_argument("--out", required=True)
    ap.add_argument("--every-s", type=float, default=1.0, help="sample one frame every N seconds")
    ap.add_argument("--min-score", type=float, default=0.25, help="pre-label threshold (low: reviewers delete)")
    a = ap.parse_args()
    zs = CFG["detector"]["zeroshot"]
    det = ZeroShotDetector(zs["model"], zs["queries"], zs["input_size"], CFG["detector"]["threads"], a.min_score)
    out = pathlib.Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    cats = [{"id": i + 1, "name": n} for i, n in enumerate(zs["queries"])]
    cat_id = {c["name"]: c["id"] for c in cats}
    images, anns = [], []
    for v in sorted(p for p in pathlib.Path(a.recordings).rglob("*") if p.suffix.lower() in VIDEO):
        with av.open(str(v)) as c:
            s = c.streams.video[0]
            nxt = 0.0
            for fr in c.decode(s):
                t = float(fr.pts * s.time_base) if fr.pts is not None else 0.0
                if t < nxt:
                    continue
                nxt = t + a.every_s
                rgb = fr.to_ndarray(format="rgb24")
                name = f"{v.stem[:40]}_{t:08.2f}.jpg"
                cv2.imwrite(str(out / "images" / name), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                img_id = len(images) + 1
                images.append({"id": img_id, "file_name": name, "width": rgb.shape[1], "height": rgb.shape[0]})
                for d in det.detect([rgb])[0]:
                    x1, y1, x2, y2 = d.box
                    anns.append({"id": len(anns) + 1, "image_id": img_id, "category_id": cat_id[d.label],
                                 "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1), "iscrowd": 0,
                                 "score": round(d.score, 3)})
        print(f"{v.name}: {len(images)} frames so far, {len(anns)} pre-labels", flush=True)
    (out / "annotations.json").write_text(json.dumps({"images": images, "annotations": anns, "categories": cats}),
                                          encoding="utf-8")
    print(f"done: {len(images)} frames, {len(anns)} pre-labels -> {out} (review ALL of them before training)")


if __name__ == "__main__":
    main()
