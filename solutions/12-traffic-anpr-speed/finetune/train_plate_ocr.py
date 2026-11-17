"""train_plate_ocr.py: train the CRNN plate reader on crops from YOUR cameras, then export to ONNX.

Data: a folder of plate crops + labels.csv (file,text) with the plate text as it should be normalised
(ASCII digits + Latin capitals, e.g. Arabic-Indic digits already converted). Collect crops automatically with the
running system (the detector saves plate crops) and label them in a spreadsheet or Label Studio.

    python finetune/train_plate_ocr.py --data data/ocr/train --val data/ocr/val --epochs 60
    -> models/ocr/plate_ocr.onnx (+ .json with the alphabet)
CPU is fine: the model has ~1M parameters; 20,000 crops x 60 epochs is a few hours on 8 Power10 cores.

Expect a plateau: CTC models first learn to output only "blank" and the loss sits near 2.5 for the first few
hundred optimiser steps, then drops sharply (measured: plateau until ~step 320, half of the training crops read
exactly by step 480). Judge a run after at least ~2,000 steps (steps = epochs x crops / batch), not after epoch 3.
"""
import argparse
import csv
import json
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parents[1] / "common" / "python"))
from power_runtime import configure_threads  # noqa: E402

configure_threads()

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from ocr import H, W, build_model, ctc_greedy, preprocess  # noqa: E402
from plates import ALPHABET, normalise  # noqa: E402


def load(folder):
    folder = pathlib.Path(folder)
    with open(folder / "labels.csv", encoding="utf-8") as f:
        rows = [(folder / r["file"], normalise(r["text"])) for r in csv.DictReader(f)]
    return [(p, t) for p, t in rows if t and p.exists()]


def augment(img):
    """CCTV-realistic: small rotation/shear, blur, brightness, noise."""
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), random.uniform(-6, 6), random.uniform(0.9, 1.1))
    m[0, 1] += random.uniform(-0.15, 0.15)
    img = cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if random.random() < 0.3:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    img = np.clip(img.astype(np.float32) * random.uniform(0.6, 1.4) + np.random.normal(0, 6, img.shape), 0, 255)
    return img.astype(np.uint8)


class Crops(torch.utils.data.Dataset):
    def __init__(self, rows, train):
        self.rows, self.train = rows, train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        path, text = self.rows[i]
        rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        if self.train:
            rgb = augment(rgb)
        return torch.from_numpy(preprocess(rgb)[0]), torch.tensor([ALPHABET.index(c) + 1 for c in text])


def collate(batch):
    x = torch.stack([b[0] for b in batch])
    targets = torch.cat([b[1] for b in batch])
    lengths = torch.tensor([len(b[1]) for b in batch])
    return x, targets, lengths


def accuracy(model, rows):
    model.eval()
    ok = 0
    with torch.no_grad():
        for path, text in rows:
            rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
            lp = model(torch.from_numpy(preprocess(rgb)))[0].numpy()
            ok += ctc_greedy(lp)[0] == text
    return ok / max(1, len(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default=str(ROOT / "models" / "ocr" / "plate_ocr.onnx"))
    a = ap.parse_args()
    train, val = load(a.data), load(a.val)
    print(f"train crops={len(train)} val crops={len(val)}")
    model = build_model()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, epochs=a.epochs,
                                                steps_per_epoch=max(1, (len(train) + a.batch - 1) // a.batch))
    ctc = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    dl = torch.utils.data.DataLoader(Crops(train, True), batch_size=a.batch, shuffle=True, collate_fn=collate,
                                     num_workers=2)
    best, best_state = -1.0, None
    for epoch in range(a.epochs):
        model.train()
        total = 0.0
        for x, targets, lengths in dl:
            lp = model(x).permute(1, 0, 2)                         # CTC wants [T, B, C]
            loss = ctc(lp, targets, torch.full((x.shape[0],), lp.shape[0], dtype=torch.long), lengths)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            total += loss.item()
        acc = accuracy(model, val)
        print(f"epoch {epoch + 1}: loss {total / max(1, len(dl)):.3f}  val exact-match {acc:.3f}")
        if acc > best:
            best, best_state = acc, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    x = torch.rand(1, 1, H, W)
    torch.onnx.export(model, (x,), str(out), input_names=["image"], output_names=["logprobs"], opset_version=18,
                      dynamo=True, external_data=False)
    import onnxruntime as ort
    ref = model(x).detach().numpy()
    got = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"]).run(None, {"image": x.numpy()})[0]
    diff = float(np.abs(ref - got).max())
    if diff > 1e-3:
        raise SystemExit(f"ONNX parity failed: {diff}")
    out.with_suffix(".json").write_text(json.dumps({"alphabet": ALPHABET, "val_exact_match": best, "input": [H, W]}))
    print(f"best val exact-match {best:.3f}; ONNX parity {diff:.1e} -> {out}")


if __name__ == "__main__":
    main()
