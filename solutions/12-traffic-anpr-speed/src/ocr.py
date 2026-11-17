"""ocr.py: licence-plate recogniser. A compact CRNN (CNN + BiLSTM) trained with CTC loss.

Why a small purpose-built model instead of a general OCR model: plates are one short line of a fixed alphabet,
and a ~1M-parameter CRNN reads a 160x48 crop in about a millisecond on one CPU core. It is trained on crops from
YOUR cameras (finetune/train_plate_ocr.py), which is what makes it accurate on Iraqi plates.
"""
import json
import pathlib

import numpy as np

from plates import ALPHABET

H, W = 48, 160  # input crop size (grey)


def build_model():
    import torch
    from torch import nn

    class CRNN(nn.Module):
        def __init__(self, n_classes=len(ALPHABET) + 1):  # +1 = CTC blank (index 0)
            super().__init__()

            def block(i, o, pool):
                return [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True), nn.MaxPool2d(pool)]
            self.cnn = nn.Sequential(*block(1, 32, (2, 2)), *block(32, 64, (2, 2)), *block(64, 128, (2, 1)),
                                     *block(128, 128, (2, 1)), nn.Conv2d(128, 192, (3, 1)), nn.ReLU(inplace=True))
            self.rnn = nn.LSTM(192, 128, num_layers=2, bidirectional=True, batch_first=True, dropout=0.1)
            self.fc = nn.Linear(256, n_classes)

        def forward(self, x):                      # x: [B, 1, 48, 160] in 0..1
            f = self.cnn(x)                        # [B, 192, 1, 40]
            f = f.squeeze(2).permute(0, 2, 1)      # [B, 40, 192]: 40 time steps left -> right
            out, _ = self.rnn(f)
            return self.fc(out).log_softmax(-1)    # [B, 40, classes]
    return CRNN()


def preprocess(rgb_crop):
    import cv2
    g = cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2GRAY)
    g = cv2.resize(g, (W, H), interpolation=cv2.INTER_AREA)
    g = cv2.equalizeHist(g)                        # day/night/IR contrast normalisation
    return (g.astype(np.float32) / 255.0)[None, None]


def ctc_greedy(logprobs):
    """logprobs: [T, C] -> (text, confidence). Confidence = product of the max probabilities of kept characters."""
    best = logprobs.argmax(-1)
    probs = np.exp(logprobs.max(-1))
    text, conf, prev = [], 1.0, 0
    for t, k in enumerate(best):
        if k != 0 and k != prev:
            text.append(ALPHABET[k - 1])
            conf *= float(probs[t])
        prev = k
    return "".join(text), conf if text else 0.0


class OnnxPlateReader:
    def __init__(self, path, threads=1):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        meta = pathlib.Path(path).with_suffix(".json")
        if meta.exists() and json.loads(meta.read_text())["alphabet"] != ALPHABET:
            raise ValueError("OCR model was trained with a different alphabet")

    def read(self, rgb_crop):
        lp = self.session.run(None, {"image": preprocess(rgb_crop)})[0][0]
        return ctc_greedy(lp)
