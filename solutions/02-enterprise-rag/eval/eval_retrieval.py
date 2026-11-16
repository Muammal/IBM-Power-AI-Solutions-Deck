"""eval_retrieval.py: how often does the right chunk come back? (Recall@k and MRR)

    python eval/eval_retrieval.py                              # uses data/pairs_eval.jsonl
    python eval/eval_retrieval.py --pairs data/eval_questions.jsonl
Each line needs "question" and "chunk_id" (or "source", matched against the chunk's file).
Run it before and after fine-tuning; keep the model that wins. Typical targets: Recall@5 >= 0.85.
"""
import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rag import RAG  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=str(ROOT / "data" / "pairs_eval.jsonl"))
    ap.add_argument("--k", type=int, default=5)
    a = ap.parse_args()
    rag = RAG()
    rag.cfg["retrieval"]["min_score"] = -1.0  # measure ranking, not the threshold
    with open(a.pairs, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    hits, rr = 0, 0.0
    for r in rows:
        got = rag.retrieve(r["question"], a.k)
        keys = [g["id"] for g in got] if "chunk_id" in r else [g["source"] for g in got]
        want = r.get("chunk_id") or r["source"]
        if want in keys:
            hits += 1
            rr += 1 / (keys.index(want) + 1)
        else:
            print("MISS:", r["question"][:80])
    n = max(1, len(rows))
    print(f"\nquestions={len(rows)}  Recall@{a.k}={hits / n:.3f}  MRR@{a.k}={rr / n:.3f}")


if __name__ == "__main__":
    main()
