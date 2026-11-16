"""export_onnx.py: PyTorch classifier -> ONNX (+ optional int8 dynamic quantisation) -> parity check.

    python src/export_onnx.py              # out/classifier/model.onnx
    python src/export_onnx.py --int8       # also model.int8.onnx (smaller, faster; check accuracy)
ONNX Runtime's MLAS library has POWER10 MMA kernels for fp32 and int8 GEMM, so serving needs no torch.
"""
import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1] / "common" / "python"))
sys.path.insert(0, str(ROOT / "src"))
from power_runtime import load_yaml  # noqa: E402

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from model import TicketClassifier  # noqa: E402

CFG = load_yaml(ROOT / "config.yaml")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--int8", action="store_true")
    a = ap.parse_args()
    out = ROOT / CFG["model"]["out_dir"]
    labels = json.loads((out / "labels.json").read_text(encoding="utf-8"))
    model = TicketClassifier(labels["base"], len(labels["category"]), len(labels["sentiment"]))
    model.load_state_dict(torch.load(out / "model.pt", map_location="cpu"))
    model.eval()
    tok = AutoTokenizer.from_pretrained(out)
    enc = tok(["الإنترنت مقطوع من الصباح", "billing question"], padding=True, return_tensors="pt")

    onnx_path = ROOT / CFG["model"]["onnx"]
    torch.onnx.export(model, (enc["input_ids"], enc["attention_mask"]), str(onnx_path),
                      input_names=["input_ids", "attention_mask"], output_names=["category", "sentiment"],
                      dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
                                    "category": {0: "batch"}, "sentiment": {0: "batch"}},
                      opset_version=17, dynamo=False)  # classic exporter: no onnxscript dependency
    with torch.inference_mode():
        ref = model(enc["input_ids"], enc["attention_mask"])[0].numpy()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = sess.run(["category"], {"input_ids": enc["input_ids"].numpy(), "attention_mask": enc["attention_mask"].numpy()})[0]
    print("max |torch - onnx| =", float(np.abs(ref - got).max()), "->", onnx_path)

    if a.int8:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        q = onnx_path.with_suffix(".int8.onnx")
        quantize_dynamic(str(onnx_path), str(q), weight_type=QuantType.QInt8)
        print("int8 model:", q, f"({q.stat().st_size / 1e6:.0f} MB vs {onnx_path.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
