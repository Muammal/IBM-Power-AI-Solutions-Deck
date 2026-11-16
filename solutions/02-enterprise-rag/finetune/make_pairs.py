"""make_pairs.py: build (question, passage) training pairs from YOUR documents with Granite 4.2.

For each chunk in index/chunks.jsonl the LLM writes questions an employee would ask that the chunk answers,
in Arabic and English. Output: data/pairs.jsonl  {"question": ..., "chunk_id": ..., "passage": ...}
A random 10% goes to data/pairs_eval.jsonl for eval/eval_retrieval.py.

    python finetune/make_pairs.py --per-chunk 3
Review a sample by hand before training: bad synthetic questions teach the embedder bad habits.
"""
import argparse
import json
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1] / "common" / "python"))
from llm_client import LLM  # noqa: E402
from power_runtime import load_yaml  # noqa: E402

SCHEMA = {"type": "object", "properties": {"questions": {"type": "array", "items": {"type": "string"},
                                                         "minItems": 1, "maxItems": 6}},
          "required": ["questions"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-chunk", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="only the first N chunks (0 = all)")
    a = ap.parse_args()
    cfg = load_yaml(ROOT / "config.yaml")
    llm = LLM(cfg["llm"]["base_url"])
    with open(ROOT / cfg["corpus"]["index_dir"] / "chunks.jsonl", encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]
    if a.limit:
        chunks = chunks[: a.limit]
    random.seed(7)
    out_train = open(ROOT / "data" / "pairs.jsonl", "w", encoding="utf-8")
    out_eval = open(ROOT / "data" / "pairs_eval.jsonl", "w", encoding="utf-8")
    n = 0
    for c in chunks:
        prompt = (f"Write {a.per_chunk} different questions that an employee could ask and that this passage "
                  f"answers. Mix Modern Standard Arabic, Iraqi Arabic and English. Questions must be answerable "
                  f"from the passage alone. Return JSON.\n\nPassage:\n{c['text']}")
        try:
            qs = llm.chat([{"role": "user", "content": prompt}], json_schema=SCHEMA, temperature=0.7)["questions"]
        except Exception as exc:  # keep going: one bad chunk should not stop a long run
            print("skip", c["id"], exc)
            continue
        for q in qs:
            row = json.dumps({"question": q.strip(), "chunk_id": c["id"], "passage": c["text"]}, ensure_ascii=False)
            (out_eval if random.random() < 0.1 else out_train).write(row + "\n")
            n += 1
        print(f"{c['id']}: {len(qs)} questions")
    print(f"{n} pairs written to data/pairs.jsonl and data/pairs_eval.jsonl")


if __name__ == "__main__":
    main()
