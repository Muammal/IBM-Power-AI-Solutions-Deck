"""app.py: upload a call recording, get transcript + analytics.

    python src/app.py
    curl -s -F file=@call.m4a localhost:8004/calls | jq .analytics
"""
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from pipeline import CFG, process  # noqa: E402

import httpx  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI, File, HTTPException, UploadFile  # noqa: E402

app = FastAPI(title="Arabic speech analytics on IBM Power")


@app.get("/health")
def health():
    try:
        asr = httpx.get(CFG["asr"]["url"], timeout=5).status_code < 500
    except httpx.HTTPError:
        asr = False
    return {"status": "ok", "asr": asr}


@app.post("/calls")
async def calls(file: UploadFile = File(...)):
    suffix = pathlib.Path(file.filename or "call.wav").suffix.lower()
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / f"upload{suffix}"
        path.write_bytes(await file.read())
        try:
            return process(path)
        except ValueError as exc:
            raise HTTPException(400, str(exc))


if __name__ == "__main__":
    uvicorn.run(app, host=CFG["server"]["host"], port=int(CFG["server"]["port"]))
