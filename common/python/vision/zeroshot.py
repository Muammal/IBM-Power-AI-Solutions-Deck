"""zeroshot.py: open-vocabulary detector (OWLv2, Apache-2.0) with the same interface as OnnxDetector.

Use it to start BEFORE you have a trained site model: classes are text queries ("a photo of a handgun"), no training
needed. Trade-offs (measured 2026-11-16 on an x86 laptop, 8 threads; measure on your LPAR):
  * accuracy: weapon scores 0.50-0.72 on firearms-range footage vs <= 0.34 on street/mall/parking footage at 640 px
  * speed: ~1.3 s per frame at 640 px (4.8 s at the native 960 px): a few cameras at 0.5 analysed fps per LPAR
Production path: run it as a TEACHER over your recordings (finetune/autolabel.py), have people verify the boxes,
then train the fast RT-DETRv2 student (train_rtdetr.py) and switch detector.backend back to "onnx".

Why OWLv2 and not OmDet-Turbo / Grounding DINO: OWLv2 needs only transformers + torch. OmDet-Turbo additionally needs
torchvision + timm and failed to run with current timm (2026-11-16).
"""
import torch
from transformers import AutoProcessor, Owlv2ForObjectDetection

from .detector import Detection, nms


class ZeroShotDetector:
    def __init__(self, model_id, queries, input_size=640, threads=0, min_score=0.15):
        """queries: {label: [prompt, ...]} e.g. {"handgun": ["a photo of a handgun", "a photo of a pistol"]}"""
        if threads:
            torch.set_num_threads(int(threads))
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.proc.image_processor.size = {"height": input_size, "width": input_size}
        self.model = Owlv2ForObjectDetection.from_pretrained(model_id).eval()
        self.prompts, self.prompt_label = [], []
        for label, texts in queries.items():
            for t in texts:
                self.prompts.append(t)
                self.prompt_label.append(label)
        self.labels = list(queries)
        self.min_score = min_score

    @torch.inference_mode()
    def detect(self, frames, keep_labels=None):
        out = []
        for f in frames:
            h, w = f.shape[:2]
            inp = self.proc(text=[self.prompts], images=f, return_tensors="pt")
            res = self.model(**inp, interpolate_pos_encoding=True)
            # OWLv2 pads the image to a square (bottom/right) before resizing: boxes are relative to that square.
            s = max(h, w)
            r = self.proc.post_process_grounded_object_detection(res, threshold=self.min_score,
                                                                 target_sizes=[(s, s)])[0]
            dets = []
            for score, lab, box in zip(r["scores"].tolist(), r["labels"].tolist(), r["boxes"].tolist()):
                label = self.prompt_label[lab]
                if keep_labels and label not in keep_labels:
                    continue
                x1, y1, x2, y2 = box
                dets.append(Detection(label, float(score), (max(0.0, x1), max(0.0, y1), min(float(w), x2), min(float(h), y2))))
            out.append(nms(dets, 0.5))          # synonyms ("handgun"/"pistol") collapse into one box
        return out
