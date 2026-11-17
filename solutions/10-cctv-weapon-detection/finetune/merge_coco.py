"""merge_coco.py: merge COCO datasets with the same categories into a new folder (ids renumbered).

    python finetune/merge_coco.py data/train data/training/from_reviews data/train_v2
"""
import json
import pathlib
import shutil
import sys


def main():
    *sources, target = [pathlib.Path(p) for p in sys.argv[1:]]
    (target / "images").mkdir(parents=True, exist_ok=True)
    images, anns, cats = [], [], None
    for src in sources:
        coco = json.loads((src / "annotations.json").read_text(encoding="utf-8"))
        if cats is None:
            cats = coco["categories"]
        elif {c["name"]: c["id"] for c in coco["categories"]} != {c["name"]: c["id"] for c in cats}:
            raise SystemExit(f"{src}: categories differ from the first dataset")
        id_map = {}
        for im in coco["images"]:
            name = f"{src.name}__{pathlib.Path(im['file_name']).name}"
            shutil.copy2(src / "images" / pathlib.Path(im["file_name"]).name, target / "images" / name)
            id_map[im["id"]] = len(images) + 1
            images.append(dict(im, id=id_map[im["id"]], file_name=name))
        for an in coco["annotations"]:
            anns.append(dict(an, id=len(anns) + 1, image_id=id_map[an["image_id"]]))
    (target / "annotations.json").write_text(json.dumps({"images": images, "annotations": anns, "categories": cats}),
                                             encoding="utf-8")
    print(f"{len(images)} images, {len(anns)} boxes -> {target}")


if __name__ == "__main__":
    main()
