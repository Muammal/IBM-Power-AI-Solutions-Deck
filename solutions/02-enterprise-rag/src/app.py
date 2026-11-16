"""app.py: HTTP API for the document assistant.

    python src/app.py
    curl -s localhost:8002/ask -H 'Content-Type: application/json' -d '{"question":"ما هي سياسة كلمات المرور؟"}'
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from rag import CFG, RAG  # noqa: E402

import uvicorn  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

app = FastAPI(title="Enterprise RAG on IBM Power")
rag = RAG()


class Ask(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)


@app.get("/health")
def health():
    return {"status": "ok", "chunks": len(rag.chunks), "llm": rag.llm.health()}


@app.post("/ask")
def ask(req: Ask):
    return rag.answer(req.question, req.top_k)


@app.post("/search")
def search(req: Ask):
    """Retrieval only (no LLM): useful for debugging and for agents that want raw passages."""
    return {"hits": rag.retrieve(req.question, req.top_k)}


if __name__ == "__main__":
    uvicorn.run(app, host=CFG["server"]["host"], port=int(CFG["server"]["port"]))
