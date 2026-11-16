"""app.py: translation API.

    python src/app.py
    curl -s localhost:8008/translate -H 'Content-Type: application/json' \
         -d '{"text":"يرجى مراجعة الفرع لتحديث بياناتك", "target":"ckb"}' | jq
"""
import pathlib
import sys
import threading

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from translate import CFG, Translator  # noqa: E402

import uvicorn  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

app = FastAPI(title="AR / KU / EN translation on IBM Power")
T = Translator()
LOCK = threading.Lock()  # one generation at a time: all cores work on it (better latency than sharing)


class Req(BaseModel):
    text: str = Field(min_length=1, max_length=8000)
    target: str


@app.get("/health")
def health():
    return {"status": "ok", "model": CFG["model"]["id"], "adapter": CFG["model"]["adapter"],
            "languages": CFG["languages"]}


@app.post("/translate")
def translate(req: Req):
    with LOCK:
        return T.postedit(req.text, T.translate(req.text, req.target), req.target)


if __name__ == "__main__":
    uvicorn.run(app, host=CFG["server"]["host"], port=int(CFG["server"]["port"]))
