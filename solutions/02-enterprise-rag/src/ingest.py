"""ingest.py: read documents -> chunks -> embeddings -> index/ (vectors.npy + chunks.jsonl).

    python src/ingest.py                 # (re)build the whole index from corpus.docs_dir
The index is a plain NumPy matrix: search = one matrix-vector product, which IBM's OpenBLAS runs with
Power10 MMA. That is fast up to a few hundred thousand chunks. Beyond that use faiss-cpu or chromadb
(both have ppc64le wheels on IBM's rolling index).
"""
import json
import pathlib
import re
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1] / "common" / "python"))
from power_runtime import configure_threads, load_yaml  # noqa: E402

CFG = load_yaml(ROOT / "config.yaml")
configure_threads(CFG["embeddings"]["threads"])

import numpy as np  # noqa: E402

from embedder import Embedder  # noqa: E402


def read_text(path):
    ext = path.suffix.lower()
    if ext in (".md", ".txt"):
        return path.read_text(encoding="utf-8", errors="replace")
    if ext == ".pdf":
        from pypdf import PdfReader
        return "\n\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    if ext == ".docx":
        import docx
        return "\n\n".join(p.text for p in docx.Document(str(path)).paragraphs)
    return ""


def chunk(text, size, overlap):
    """Split on paragraph boundaries, then pack paragraphs into ~size-character chunks with overlap.
    Character-based on purpose: it treats Arabic and English the same and needs no tokenizer."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        while len(p) > size:  # a single huge paragraph: hard split
            chunks.append(p[:size])
            p = p[size - overlap:]
        if len(cur) + len(p) + 2 > size and cur:
            chunks.append(cur)
            cur = cur[-overlap:] + "\n\n" + p
        else:
            cur = f"{cur}\n\n{p}" if cur else p
    if cur:
        chunks.append(cur)
    return chunks


def main():
    c = CFG["corpus"]
    docs_dir, index_dir = ROOT / c["docs_dir"], ROOT / c["index_dir"]
    index_dir.mkdir(exist_ok=True)
    records = []
    for path in sorted(docs_dir.rglob("*")):
        if path.is_file():
            text = read_text(path)
            for i, ch in enumerate(chunk(text, c["chunk_chars"], c["overlap_chars"])):
                records.append({"id": f"{path.relative_to(docs_dir)}#{i}", "source": str(path.relative_to(docs_dir)),
                                "text": ch})
    print(f"{len(records)} chunks from {docs_dir}")
    e = CFG["embeddings"]
    emb = Embedder(e["model"], e["pooling"], e["max_length"], e["batch_size"])
    t = time.time()
    vecs = emb.encode([r["text"] for r in records]).astype(np.float32)
    print(f"embedded in {time.time() - t:.1f}s -> {vecs.shape}")
    np.save(index_dir / "vectors.npy", vecs)
    with open(index_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (index_dir / "meta.json").write_text(json.dumps({"model": e["model"], "pooling": e["pooling"],
                                                      "count": len(records)}), encoding="utf-8")
    print("index written to", index_dir)


if __name__ == "__main__":
    main()
