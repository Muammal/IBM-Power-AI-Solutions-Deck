"""train_classifier.py: fine-tune the 278M multilingual encoder on tickets (CPU, plain PyTorch loop).

    python finetune/train_classifier.py                       # human labels only
    python finetune/train_classifier.py --with-llm-labels     # + data/llm_labeled.csv (bootstrap)
Saves out/classifier/{model.pt, tokenizer files, labels.json} and prints a test report.
Time (8 Power10 cores, measure yours): ~5,000 tickets x 4 epochs ~ 30-60 min in bf16 autocast.
"""
import argparse
import csv
import json
import pathlib
import random
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1] / "common" / "python"))
sys.path.insert(0, str(ROOT / "src"))
from power_runtime import configure_threads, load_yaml  # noqa: E402

CFG = load_yaml(ROOT / "config.yaml")
configure_threads(CFG["runtime"]["threads"])

import torch  # noqa: E402
from sklearn.metrics import classification_report  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from model import TicketClassifier  # noqa: E402


def read(path):
    with open(path, encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r.get("text")]


def batches(rows, tok, cats, sents, size, max_len, shuffle):
    rows = rows[:]
    if shuffle:
        random.shuffle(rows)
    for i in range(0, len(rows), size):
        b = rows[i:i + size]
        enc = tok([r["text"] for r in b], padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        yield enc, torch.tensor([cats.index(r["category"]) for r in b]), torch.tensor([sents.index(r["sentiment"]) for r in b])


def predict(model, rows, tok, cats, sents, m):
    model.eval()
    pc, ps = [], []
    with torch.inference_mode():
        for enc, _, _ in batches(rows, tok, cats, sents, 64, m["max_length"], False):
            lc, ls = model(enc["input_ids"], enc["attention_mask"])
            pc += lc.argmax(-1).tolist()
            ps += ls.argmax(-1).tolist()
    return pc, ps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-llm-labels", action="store_true")
    ap.add_argument("--gpu", action="store_true", help="one CUDA device (see COMPLIANCE.md)")
    a = ap.parse_args()
    random.seed(0)
    torch.manual_seed(0)
    m, cats, sents = CFG["model"], CFG["labels"]["category"], CFG["labels"]["sentiment"]

    human = read(ROOT / CFG["data"]["labeled_csv"])
    random.shuffle(human)
    n_test = max(1, int(len(human) * CFG["data"]["test_fraction"]))
    test, train = human[:n_test], human[n_test:]       # test set = human labels only, always
    if a.with_llm_labels:
        train += read(ROOT / CFG["data"]["llm_labeled_csv"])
    print(f"train={len(train)} test={len(test)}")

    device = "cuda" if a.gpu else "cpu"
    tok = AutoTokenizer.from_pretrained(m["base"])
    model = TicketClassifier(m["base"], len(cats), len(sents)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(m["learning_rate"]), weight_decay=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()

    t0 = time.time()
    for epoch in range(int(m["epochs"])):
        model.train()
        total = 0.0
        for enc, yc, ys in batches(train, tok, cats, sents, int(m["batch_size"]), m["max_length"], True):
            with torch.autocast(device_type=device, dtype=torch.bfloat16):  # Power10 MMA has bf16 GEMM
                lc, ls = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
            loss = loss_fn(lc.float(), yc.to(device)) + 0.5 * loss_fn(ls.float(), ys.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            total += loss.item()
        print(f"epoch {epoch + 1}: loss {total:.3f} ({time.time() - t0:.0f}s)")

    model.to("cpu")
    pc, ps = predict(model, test, tok, cats, sents, m)
    print(classification_report([cats.index(r["category"]) for r in test], pc, labels=list(range(len(cats))),
                                target_names=cats, zero_division=0))
    print(classification_report([sents.index(r["sentiment"]) for r in test], ps, labels=list(range(len(sents))),
                                target_names=sents, zero_division=0))

    out = ROOT / m["out_dir"]
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    tok.save_pretrained(out)
    (out / "labels.json").write_text(json.dumps({"category": cats, "sentiment": sents, "base": m["base"]}), encoding="utf-8")
    print("saved", out, "-> next: python src/export_onnx.py")


if __name__ == "__main__":
    main()
