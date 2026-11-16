"""rag.py: retrieve the best chunks for a question and ask Granite 4.2 to answer with citations.

    python src/rag.py "كم عدد أيام الإجازة السنوية؟"
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1] / "common" / "python"))
from power_runtime import configure_threads, load_yaml  # noqa: E402

CFG = load_yaml(ROOT / "config.yaml")
configure_threads(CFG["embeddings"]["threads"])

import numpy as np  # noqa: E402

from embedder import Embedder  # noqa: E402
from llm_client import LLM  # noqa: E402


class RAG:
    def __init__(self, cfg=CFG):
        self.cfg = cfg
        idx = ROOT / cfg["corpus"]["index_dir"]
        meta = json.loads((idx / "meta.json").read_text(encoding="utf-8"))
        e = cfg["embeddings"]
        if meta["model"] != e["model"]:
            raise SystemExit(f"Index was built with {meta['model']} but config uses {e['model']}: run src/ingest.py")
        self.vecs = np.load(idx / "vectors.npy")
        with open(idx / "chunks.jsonl", encoding="utf-8") as f:
            self.chunks = [json.loads(line) for line in f]
        self.emb = Embedder(e["model"], e["pooling"], e["max_length"], e["batch_size"])
        self.llm = LLM(cfg["llm"]["base_url"])

    def retrieve(self, question, top_k=None):
        r = self.cfg["retrieval"]
        q = self.emb.encode([question])[0]
        scores = self.vecs @ q                       # cosine similarity (vectors are normalised)
        best = np.argsort(-scores)[: top_k or r["top_k"]]
        return [dict(self.chunks[i], score=float(scores[i])) for i in best if scores[i] >= r["min_score"]]

    def answer(self, question, top_k=None):
        hits = self.retrieve(question, top_k)
        if not hits:
            return {"answer": "لم أجد هذا في الوثائق المتاحة. / I could not find this in the available documents.",
                    "sources": []}
        context = "\n\n".join(f"[{i + 1}] ({h['source']})\n{h['text']}" for i, h in enumerate(hits))
        l = self.cfg["llm"]
        text = self.llm.chat(
            [{"role": "system", "content": self.cfg["prompts"]["system"].strip()},
             {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}],
            temperature=l["temperature"], max_tokens=l["max_tokens"], thinking=l["thinking"])
        return {"answer": text,
                "sources": [{"n": i + 1, "source": h["source"], "score": round(h["score"], 3)} for i, h in enumerate(hits)]}


if __name__ == "__main__":
    out = RAG().answer(" ".join(sys.argv[1:]) or "What is the password policy?")
    print(out["answer"])
    for s in out["sources"]:
        print(f"  [{s['n']}] {s['source']} (score {s['score']})")
