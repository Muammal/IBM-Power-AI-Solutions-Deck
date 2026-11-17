"""train_rtdetr.py: fine-tune RT-DETRv2 (Apache-2.0) on your own labelled images (COCO format).

Dataset layout (export from CVAT / Label Studio as "COCO 1.0"):
    <dir>/images/*.jpg
    <dir>/annotations.json        # COCO: images, annotations (bbox = x,y,w,h in pixels), categories

    python common/python/vision/train_rtdetr.py --train data/train --val data/val --out out/detector \
        --base PekingU/rtdetr_v2_r18vd --epochs 40 --size 640
    python common/python/vision/export_rtdetr.py --model out/detector/best --out models/detector

Compute: fine-tuning a 20M-parameter detector is the heaviest job in this repo. On CPU (Power10, 16 cores) budget
roughly a day for ~2,000 images x 40 epochs at 640 px; use --size 512 and --freeze-backbone to cut that by ~3x.
A GPU that passes COMPLIANCE.md (T4/V100/A10 class) brings it to hours: --gpu. Inference always runs on Power CPU.
"""
import argparse
import json
import pathlib
import random

import torch
from PIL import Image, ImageEnhance
from transformers import (EarlyStoppingCallback, RTDetrImageProcessor, RTDetrV2ForObjectDetection, Trainer,
                          TrainingArguments)


class CocoDetection(torch.utils.data.Dataset):
    def __init__(self, folder, processor, cat_map, augment):
        folder = pathlib.Path(folder)
        coco = json.loads((folder / "annotations.json").read_text(encoding="utf-8"))
        self.folder, self.proc, self.cat_map, self.augment = folder, processor, cat_map, augment
        self.images = coco["images"]
        self.anns = {}
        for a in coco["annotations"]:
            if a["category_id"] in cat_map:
                self.anns.setdefault(a["image_id"], []).append(a)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        info = self.images[i]
        img = Image.open(self.folder / "images" / pathlib.Path(info["file_name"]).name).convert("RGB")
        anns = [dict(bbox=list(a["bbox"]), category_id=self.cat_map[a["category_id"]], area=a["bbox"][2] * a["bbox"][3],
                     iscrowd=0) for a in self.anns.get(info["id"], [])]
        if self.augment:  # light, CCTV-realistic augmentation: mirror, brightness/contrast (day/night, IR)
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                for a in anns:
                    a["bbox"][0] = img.width - a["bbox"][0] - a["bbox"][2]
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.6, 1.4))
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.7, 1.3))
            if random.random() < 0.15:
                img = img.convert("L").convert("RGB")  # IR / night cameras are grey
        enc = self.proc(images=img, annotations={"image_id": info["id"], "annotations": anns}, return_tensors="pt")
        return {"pixel_values": enc["pixel_values"][0], "labels": enc["labels"][0]}


def collate(batch):
    return {"pixel_values": torch.stack([b["pixel_values"] for b in batch]), "labels": [b["labels"] for b in batch]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", default="out/detector")
    ap.add_argument("--base", default="PekingU/rtdetr_v2_r18vd")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--freeze-backbone", action="store_true")
    ap.add_argument("--gpu", action="store_true", help="one CUDA device (see COMPLIANCE.md)")
    a = ap.parse_args()

    cats = json.loads((pathlib.Path(a.train) / "annotations.json").read_text(encoding="utf-8"))["categories"]
    cats = sorted(cats, key=lambda c: c["id"])
    cat_map = {c["id"]: i for i, c in enumerate(cats)}
    id2label = {i: c["name"] for i, c in enumerate(cats)}
    print("classes:", id2label)

    proc = RTDetrImageProcessor.from_pretrained(a.base, size={"height": a.size, "width": a.size})
    model = RTDetrV2ForObjectDetection.from_pretrained(a.base, id2label=id2label,
                                                       label2id={v: k for k, v in id2label.items()},
                                                       ignore_mismatched_sizes=True)
    if a.freeze_backbone:
        for p in model.model.backbone.parameters():
            p.requires_grad = False
    train = CocoDetection(a.train, proc, cat_map, augment=True)
    val = CocoDetection(a.val, proc, cat_map, augment=False)
    print(f"train images={len(train)} val images={len(val)}")

    args = TrainingArguments(
        output_dir=a.out, num_train_epochs=a.epochs, learning_rate=a.lr, weight_decay=1e-4,
        per_device_train_batch_size=a.batch, per_device_eval_batch_size=a.batch, lr_scheduler_type="cosine",
        warmup_ratio=0.05, max_grad_norm=0.1, eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        remove_unused_columns=False, dataloader_num_workers=2, use_cpu=not a.gpu, logging_steps=20, report_to=[])
    trainer = Trainer(model=model, args=args, train_dataset=train, eval_dataset=val, data_collator=collate,
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=8)])
    trainer.train()
    best = pathlib.Path(a.out) / "best"
    trainer.save_model(str(best))
    proc.save_pretrained(str(best))
    print("saved", best, "-> next: export_rtdetr.py --model", best)


if __name__ == "__main__":
    main()
