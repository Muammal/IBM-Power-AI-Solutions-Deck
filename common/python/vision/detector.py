"""detector.py: RT-DETR / RT-DETRv2 object detector on ONNX Runtime (CPU).

The ONNX file is produced by export_rtdetr.py (inputs: pixel_values [B,3,S,S] in 0..1; outputs: logits [B,Q,C],
pred_boxes [B,Q,4] normalised cx,cy,w,h). labels.json next to the ONNX file lists the class names.
ONNX Runtime's MLAS kernels use POWER10 MMA for the convolutions and matrix multiplies.
"""
import dataclasses
import json
import pathlib

import numpy as np
import onnxruntime as ort


@dataclasses.dataclass
class Detection:
    label: str
    score: float
    box: tuple  # x1, y1, x2, y2 in pixels of the original frame


class OnnxDetector:
    def __init__(self, onnx_path, input_size=640, threads=0, min_score=0.25):
        onnx_path = pathlib.Path(onnx_path)
        meta = json.loads((onnx_path.parent / "labels.json").read_text(encoding="utf-8"))
        self.labels = meta["labels"]
        self.input_size = int(meta.get("input_size", input_size))
        self.min_score = min_score
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(threads or 0)
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])

    def preprocess(self, frames):
        import cv2
        s = self.input_size
        batch = np.stack([cv2.resize(f, (s, s), interpolation=cv2.INTER_LINEAR) for f in frames]).astype(np.float32)
        return np.ascontiguousarray(batch.transpose(0, 3, 1, 2) / 255.0)

    def detect(self, frames, keep_labels=None):
        """frames: list of RGB uint8 arrays (any size). Returns one list of Detection per frame."""
        if not frames:
            return []
        # The exported graph has batch size 1 (see export_rtdetr.py): run frame by frame. Each call already uses
        # every core through ONNX Runtime's intra-op threads, so this is as fast as batching on CPU.
        outs = [self.session.run(None, {"pixel_values": self.preprocess([f])}) for f in frames]
        logits = np.concatenate([o[0] for o in outs])
        boxes = np.concatenate([o[1] for o in outs])
        probs = 1.0 / (1.0 + np.exp(-logits))                       # RT-DETR uses sigmoid (focal loss), not softmax
        out = []
        for f, p, b in zip(frames, probs, boxes):
            h, w = f.shape[:2]
            cls, score = p.argmax(-1), p.max(-1)
            dets = []
            for q in np.nonzero(score >= self.min_score)[0]:
                label = self.labels[int(cls[q])]
                if keep_labels and label not in keep_labels:
                    continue
                cx, cy, bw, bh = b[q]
                box = (max(0.0, (cx - bw / 2) * w), max(0.0, (cy - bh / 2) * h),
                       min(float(w), (cx + bw / 2) * w), min(float(h), (cy + bh / 2) * h))
                dets.append(Detection(label, float(score[q]), tuple(round(v, 1) for v in box)))
            out.append(nms(dets, 0.6))
        return out


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def nms(dets, thr):
    """RT-DETR rarely duplicates boxes, but a light per-class NMS keeps alerts clean."""
    keep = []
    for d in sorted(dets, key=lambda d: -d.score):
        if all(k.label != d.label or iou(k.box, d.box) < thr for k in keep):
            keep.append(d)
    return keep
