"""build_dataset.py: turn operator reviews into training data (the continuous-improvement loop).

  * confirmed alerts  -> positive examples: raw frame + the alerted box as a weapon label (VERIFY boxes in CVAT)
  * false alarms      -> hard negatives: raw frame with NO weapon box (teaches the model "a phone is not a gun")
Output: data/training/from_reviews/{images/, annotations.json} in COCO format, using the class ids of your
existing training set so the two can be merged:

    python finetune/build_dataset.py --classes-from data/train/annotations.json
    python finetune/merge_coco.py data/train data/training/from_reviews data/train_v2
"""
import argparse
import datetime as dt
import json
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import cv2  # noqa: E402

from app import CFG  # noqa: E402
from events import EventStore  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes-from", required=True, help="annotations.json of the current training set")
    ap.add_argument("--out", default=str(ROOT / "data" / "training" / "from_reviews"))
    ap.add_argument("--since-days", type=int, default=90)
    a = ap.parse_args()
    cats = json.loads(pathlib.Path(a.classes_from).read_text(encoding="utf-8"))["categories"]
    cat_id = {c["name"]: c["id"] for c in cats}
    store = EventStore(ROOT / CFG["storage"]["db"])
    media = ROOT / CFG["storage"]["media_dir"]
    out = pathlib.Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    since = dt.datetime.now().timestamp() - a.since_days * 86400
    images, anns, n_pos, n_neg = [], [], 0, 0
    for ev in store.q("SELECT * FROM events WHERE status IN ('confirmed','false_alarm') AND ts>=?", (since,)):
        raw = media / f"{ev['id']}_raw.jpg"
        if not raw.exists():
            continue
        h, w = cv2.imread(str(raw)).shape[:2]
        img_id = len(images) + 1
        shutil.copy2(raw, out / "images" / raw.name)
        images.append({"id": img_id, "file_name": raw.name, "width": w, "height": h})
        if ev["status"] == "confirmed" and ev["label"] in cat_id:
            x1, y1, x2, y2 = json.loads(ev["box"])
            anns.append({"id": len(anns) + 1, "image_id": img_id, "category_id": cat_id[ev["label"]],
                         "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1), "iscrowd": 0})
            n_pos += 1
        else:
            n_neg += 1
    (out / "annotations.json").write_text(json.dumps({"images": images, "annotations": anns, "categories": cats},
                                                     ensure_ascii=False), encoding="utf-8")
    print(f"{n_pos} confirmed positives, {n_neg} hard negatives -> {out}")
    print("Open the positives in CVAT, correct boxes and add person boxes, then merge with your training set.")


if __name__ == "__main__":
    main()
