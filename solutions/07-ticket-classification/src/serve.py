"""serve.py: classify and route tickets with ONNX Runtime (no torch at serving time).

    python src/serve.py
    curl -s localhost:8007/classify -H 'Content-Type: application/json' -d '{"text":"ماكو انترنت بالكرادة"}' | jq
Set ONNX_MODEL=out/classifier/model.int8.onnx to serve the quantised model.
"""
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1] / "common" / "python"))
from power_runtime import configure_threads, load_yaml  # noqa: E402

CFG = load_yaml(ROOT / "config.yaml")
THREADS = configure_threads(CFG["runtime"]["threads"])

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

OUT = ROOT / CFG["model"]["out_dir"]
LABELS = json.loads((OUT / "labels.json").read_text(encoding="utf-8"))
TOK = AutoTokenizer.from_pretrained(OUT)
opts = ort.SessionOptions()
opts.intra_op_num_threads = THREADS or 0
MODEL_PATH = os.getenv("ONNX_MODEL", str(ROOT / CFG["model"]["onnx"]))
SESSION = ort.InferenceSession(MODEL_PATH, opts, providers=["CPUExecutionProvider"])
app = FastAPI(title="Ticket classification on IBM Power")


def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def classify(texts):
    enc = TOK(texts, padding=True, truncation=True, max_length=CFG["model"]["max_length"], return_tensors="np")
    cat, sen = SESSION.run(None, {"input_ids": enc["input_ids"].astype(np.int64),
                                  "attention_mask": enc["attention_mask"].astype(np.int64)})
    pc, ps = softmax(cat), softmax(sen)
    out = []
    for i, t in enumerate(texts):
        c = LABELS["category"][int(pc[i].argmax())]
        out.append({"text": t, "category": c, "category_confidence": round(float(pc[i].max()), 3),
                    "sentiment": LABELS["sentiment"][int(ps[i].argmax())],
                    "queue": CFG["routing"].get(c, "customer-care"),
                    "needs_human": bool(pc[i].max() < 0.6)})
    return out


class Req(BaseModel):
    text: str = Field(min_length=1, max_length=5000)


class BatchReq(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=256)


@app.get("/health")
def health():
    return {"status": "ok", "model": pathlib.Path(MODEL_PATH).name}


@app.post("/classify")
def classify_one(req: Req):
    return classify([req.text])[0]


@app.post("/classify_batch")
def classify_batch(req: BatchReq):
    return classify(req.texts)


if __name__ == "__main__":
    uvicorn.run(app, host=CFG["server"]["host"], port=int(CFG["server"]["port"]))
