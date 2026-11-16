"""finetune_embeddings.py: adapt the embedding model to your documents (contrastive learning, CPU).

Loss = multiple-negatives ranking (InfoNCE): in a batch of (question, passage) pairs, each question must score
its own passage higher than the other passages in the batch. That is the loss sentence-transformers calls
MultipleNegativesRankingLoss, written out in ~10 lines so you can see exactly what happens.

    python finetune/finetune_embeddings.py --epochs 1 --batch 32
    # then in config.yaml: embeddings.model: out/embed-finetuned   and re-run src/ingest.py
Time on 8 Power10 cores (measure yours): ~5,000 pairs x 1 epoch with the 278M model ~ 30-60 minutes.
"""
import argparse
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
configure_threads(CFG["embeddings"]["threads"])

import torch  # noqa: E402

from embedder import Embedder  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=str(ROOT / "data" / "pairs.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "out" / "embed-finetuned"))
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=32, help="bigger = more in-batch negatives = better, more RAM")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--scale", type=float, default=20.0, help="1/temperature for the similarity logits")
    ap.add_argument("--gpu", action="store_true", help="one CUDA device (see COMPLIANCE.md)")
    a = ap.parse_args()

    with open(a.pairs, encoding="utf-8") as f:
        pairs = [json.loads(line) for line in f if line.strip()]
    print(f"{len(pairs)} training pairs")
    e = CFG["embeddings"]
    emb = Embedder(e["model"], e["pooling"], e["max_length"])
    device = "cuda" if a.gpu else "cpu"
    emb.model.to(device).train()
    opt = torch.optim.AdamW(emb.model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * (len(pairs) // a.batch)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, steps // 10)))

    step, t0 = 0, time.time()
    for epoch in range(a.epochs):
        random.shuffle(pairs)
        for i in range(0, len(pairs) - a.batch + 1, a.batch):
            batch = pairs[i:i + a.batch]
            # bf16 autocast on CPU: Power10 MMA has native bf16 outer-product instructions
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                q = emb.forward([p["question"] for p in batch])
                d = emb.forward([p["passage"] for p in batch])
            logits = (q.float() @ d.float().T) * a.scale          # [B, B]: row i should peak at column i
            loss = torch.nn.functional.cross_entropy(logits, torch.arange(len(batch), device=logits.device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(emb.model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            step += 1
            if step % 10 == 0:
                print(f"epoch {epoch} step {step}/{steps} loss {loss.item():.4f} ({time.time() - t0:.0f}s)")

    emb.model.save_pretrained(a.out)
    emb.tok.save_pretrained(a.out)
    print("saved", a.out, "-> set embeddings.model to this path and run src/ingest.py")


if __name__ == "__main__":
    main()
