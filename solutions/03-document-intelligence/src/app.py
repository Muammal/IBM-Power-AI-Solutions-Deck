"""app.py: upload a document, get validated JSON back.

    python src/app.py
    curl -s -F file=@data/samples/invoice_001.md localhost:8003/extract | jq
"""
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from extract import CFG, extract  # noqa: E402

import uvicorn  # noqa: E402
from fastapi import FastAPI, File, HTTPException, UploadFile  # noqa: E402

app = FastAPI(title="Document intelligence on IBM Power")
MAX_BYTES = 25 * 1024 * 1024


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/extract")
async def extract_endpoint(file: UploadFile = File(...), document_type: str | None = None):
    suffix = pathlib.Path(file.filename or "upload.pdf").suffix.lower()
    body = await file.read()
    if len(body) > MAX_BYTES:
        raise HTTPException(413, "file larger than 25 MB")
    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(body)
        tmp.flush()
        result = extract(tmp.name, document_type)
    result["file"] = file.filename
    result.pop("text")
    return result


if __name__ == "__main__":
    uvicorn.run(app, host=CFG["server"]["host"], port=int(CFG["server"]["port"]))
