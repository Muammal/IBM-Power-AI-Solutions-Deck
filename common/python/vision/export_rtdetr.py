"""export_rtdetr.py: Hugging Face RT-DETR/RT-DETRv2 checkpoint -> ONNX + labels.json, with a parity check.

    python common/python/vision/export_rtdetr.py --model out/detector/best --out models/detector [--image sample.jpg]
    python common/python/vision/export_rtdetr.py --model PekingU/rtdetr_v2_r18vd --out models/coco   # COCO baseline

Parity is checked on a real (or structured) image, order-independently: sorted top scores + confident detections
matched box-to-box (RT-DETR selects queries with top-k, so near-tied queries may legitimately swap positions).
Never deploy an ONNX file whose parity check failed.
"""
import argparse
import json
import pathlib

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from transformers import AutoModelForObjectDetection


class Wrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, pixel_values):
        o = self.m(pixel_values=pixel_values)
        return o.logits, o.pred_boxes


def parity(p_ref, b_ref, p_onnx, b_onnx, top=50, det_thr=0.2, tol=2e-3):
    """Order-independent comparison of two RT-DETR outputs (one image).
    1. the sorted top-`top` query scores must match (catches numeric drift, ignores query re-ordering)
    2. every confident PyTorch detection must have an ONNX detection with the same class, IoU > 0.95, same score."""
    problems = []
    s_ref, s_onx = np.sort(p_ref.max(-1))[::-1][:top], np.sort(p_onnx.max(-1))[::-1][:top]
    d = float(np.abs(s_ref - s_onx).max())
    print(f"parity: max |sorted top-{top} score diff| = {d:.2e}")
    if d > tol:
        problems.append(f"score distribution differs by {d:.3e}")

    def xyxy(b):
        return (b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2)

    def iou(a, b):
        ix, iy = max(0, min(a[2], b[2]) - max(a[0], b[0])), max(0, min(a[3], b[3]) - max(a[1], b[1]))
        u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
        return ix * iy / u if u > 0 else 0

    confident = np.nonzero(p_ref.max(-1) >= det_thr)[0]
    for q in confident:
        c, s, box = p_ref[q].argmax(), p_ref[q].max(), xyxy(b_ref[q])
        ok = any(p_onnx[k].argmax() == c and abs(p_onnx[k].max() - s) < tol and iou(box, xyxy(b_onnx[k])) > 0.95
                 for k in np.nonzero(p_onnx.max(-1) >= det_thr - tol)[0])
        if not ok:
            problems.append(f"detection class={c} score={s:.3f} not reproduced by ONNX")
    print(f"parity: {len(confident)} confident detections compared")
    return problems


def sample_image(path, size):
    if path:
        img = np.asarray(Image.open(path).convert("RGB").resize((size, size)), np.float32) / 255.0
    else:  # structured synthetic scene: gradients + blocks
        img = np.zeros((size, size, 3), np.float32)
        img[..., 0] = np.linspace(0, 1, size)[None]
        img[size // 3: size // 3 * 2, size // 6: size // 2] = 0.8
        img[size // 2:, size // 2:, 2] = 0.6
    return torch.from_numpy(img).permute(2, 0, 1)[None].contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--image", help="a representative camera frame for the parity check")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    model = AutoModelForObjectDetection.from_pretrained(a.model).eval()
    x = sample_image(a.image, a.size)
    onnx_path = out / "model.onnx"
    # dynamo=True (torch.export based) is REQUIRED for RT-DETRv2: the classic TorchScript exporter (dynamo=False)
    # produces a graph whose scores differ by ~0.1 from PyTorch (verified 2026-11-16, torch 2.10); the parity check
    # below catches that. The exporter version matters too: onnxscript 0.5.6 also gives scores ~0.19 off, 0.7.2
    # matches to ~5e-6 on x86 and Power10 (pinned in common/requirements-vision-train.txt). Batch size is fixed at 1: the model code uses the batch size as a Python int, so it
    # cannot be exported as dynamic. On CPU this costs nothing measurable: ONNX Runtime already spreads one frame
    # over all cores (measured: batch 4 gave the same ms/frame as batch 1).
    torch.onnx.export(Wrapper(model), (x,), str(onnx_path), input_names=["pixel_values"],
                      output_names=["logits", "pred_boxes"], opset_version=18, dynamo=True, external_data=False)
    labels = [model.config.id2label[i] for i in range(len(model.config.id2label))]
    (out / "labels.json").write_text(json.dumps({"labels": labels, "input_size": a.size, "source": str(a.model)},
                                                ensure_ascii=False, indent=1), encoding="utf-8")

    with torch.inference_mode():
        ref_logits, ref_boxes = Wrapper(model)(x)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    logits, boxes = sess.run(None, {"pixel_values": x.numpy()})
    problems = parity(torch.sigmoid(ref_logits).numpy()[0], ref_boxes.numpy()[0], 1 / (1 + np.exp(-logits[0])), boxes[0])
    if problems:
        raise SystemExit("PARITY FAILED: do not deploy this ONNX file\n  " + "\n  ".join(problems))
    print(f"OK -> {onnx_path} ({onnx_path.stat().st_size / 1e6:.0f} MB), classes: {labels}")


if __name__ == "__main__":
    main()
