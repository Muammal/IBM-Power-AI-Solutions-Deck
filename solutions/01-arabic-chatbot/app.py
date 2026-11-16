"""app.py: web UI (Step 07). FastAPI serves one HTML page and streams answers.

    python app.py                    then open http://127.0.0.1:7860

Why FastAPI and not Gradio: the IBM ppc64le wheel snapshot used here (linux-v2026.03.31)
has no Gradio build, and FastAPI + uvicorn are already validated in the ppc64le/pyeco
vllm example. One small HTML page is all a chat needs.
"""
import os
import asyncio
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool

import llm

PAGE = Path(__file__).parent / "static" / "index.html"
CFG = llm.load_config()
# A CPU has one set of cores: answering two users at the same time makes both slow.
# This lock makes requests take turns. Increase capacity with more LPARs, not threads.
MODEL_LOCK = asyncio.Lock()


class Message(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")   # the browser may not send "system"
    content: str = Field(max_length=8000)


class ChatRequest(BaseModel):
    messages: list[Message] = Field(min_length=1, max_length=100)


@asynccontextmanager
async def lifespan(app):
    print("Loading model ...", flush=True)
    llm.load_model(CFG)                 # load once at startup, not per request
    print("Ready:", llm.describe(), flush=True)
    yield
    llm.unload()


app = FastAPI(title="Power chatbot", lifespan=lifespan)


@app.get("/")
def index():
    return FileResponse(PAGE)


@app.get("/health")
def health():
    return {"status": "ok", "model": llm.describe(), "busy": MODEL_LOCK.locked()}


@app.post("/chat")
async def chat(req: ChatRequest, request: Request):
    history = [m.model_dump() for m in req.messages]
    if history[-1]["role"] != "user":
        raise HTTPException(400, "The last message must come from the user.")

    async def stream():
        stop = threading.Event()     # tells llm.generate to end the answer early
        async with MODEL_LOCK:
            pieces = llm.generate(history, stream=True, stop=stop)
            try:
                # generate() blocks between tokens, so each step runs in a worker thread
                async for piece in iterate_in_threadpool(pieces):
                    if await request.is_disconnected():   # Stop button / closed tab
                        break
                    yield piece
            finally:                 # also runs if the server cancels this response
                stop.set()           # the model stops at its next token...
                try:
                    await run_in_threadpool(pieces.close)   # ...and its thread is joined
                except ValueError:   # "generator already executing": it ends by itself
                    pass

    # Plain UTF-8 text chunks. X-Accel-Buffering stops nginx-style proxies from buffering.
    return StreamingResponse(stream(), media_type="text/plain; charset=utf-8",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


if __name__ == "__main__":
    server = CFG.get("server") or {}
    host = os.getenv("CHATBOT_HOST", server.get("host", "127.0.0.1"))
    port = int(os.getenv("CHATBOT_PORT", server.get("port", 7860)))
    uvicorn.run(app, host=host, port=port)
