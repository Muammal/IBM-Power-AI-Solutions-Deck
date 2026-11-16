"""pipeline.py: audio -> transcript (whisper.cpp) -> redaction -> call analytics (Granite 4.2) -> JSON.

    python src/pipeline.py data/sample_calls/call1.wav
    python src/pipeline.py /data/recordings/2026-11-14/      # a whole folder
"""
import datetime as dt
import hashlib
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1] / "common" / "python"))
sys.path.insert(0, str(ROOT / "src"))
import httpx  # noqa: E402

from audio import to_pcm16k, to_wav_bytes  # noqa: E402
from llm_client import LLM  # noqa: E402
from power_runtime import load_yaml  # noqa: E402

CFG = load_yaml(ROOT / "config.yaml")
AUDIO_EXT = {".wav", ".mp3", ".m4a", ".ogg", ".opus", ".flac", ".aac", ".mp4", ".amr", ".webm"}

ANALYTICS_SCHEMA = {
    "type": "object",
    "properties": {
        "summary_ar": {"type": "string", "description": "2-3 sentences, Modern Standard Arabic"},
        "summary_en": {"type": "string", "description": "2-3 sentences, English"},
        "topic": {"type": "string", "enum": ["billing", "technical_issue", "new_service", "complaint",
                                             "account_change", "information", "other"]},
        "customer_sentiment": {"type": "string", "enum": ["positive", "neutral", "negative", "angry"]},
        "resolved": {"type": "boolean"},
        "action_items": {"type": "array", "items": {"type": "string"}},
        "escalate": {"type": "boolean", "description": "true if a supervisor must call back"},
    },
    "required": ["summary_ar", "summary_en", "topic", "customer_sentiment", "resolved", "action_items", "escalate"],
}

# Personal data patterns (Iraqi mobile numbers, 16-digit cards, IBAN-like). Extend for your data.
_REDACT = [
    (re.compile(r"(?:\+?964|0)\s?7[0-9](?:[\s-]?\d){8}"), "[PHONE]"),
    (re.compile(r"\b(?:\d[\s-]?){15}\d\b"), "[CARD]"),
    (re.compile(r"\bIQ\d{2}[A-Z0-9]{4}\d{15}\b"), "[IBAN]"),
]


_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


def redact(text):
    text = text.translate(_DIGITS)  # Arabic-Indic digits -> 0-9 so the patterns match
    for pattern, repl in _REDACT:
        text = pattern.sub(repl, text)
    return text


def transcribe(path):
    pcm, seconds = to_pcm16k(str(path))
    if seconds > CFG["asr"]["max_seconds"]:
        raise ValueError(f"audio is {seconds:.0f}s, limit is {CFG['asr']['max_seconds']}s")
    r = httpx.post(f"{CFG['asr']['url']}/inference", timeout=3600,
                   files={"file": ("audio.wav", to_wav_bytes(pcm), "audio/wav")},
                   data={"language": CFG["asr"]["language"], "response_format": "json", "temperature": "0.0"})
    r.raise_for_status()
    return r.json()["text"].strip(), seconds


def analyse(transcript, llm=None):
    llm = llm or LLM(CFG["llm"]["base_url"])
    prompt = ("This is an automatic transcript of a customer-service phone call in Iraqi Arabic or English. "
              "It may contain recognition errors. Analyse it and return JSON.\n\nTranscript:\n" + transcript[:20000])
    return llm.chat([{"role": "user", "content": prompt}], json_schema=ANALYTICS_SCHEMA,
                    temperature=CFG["llm"]["temperature"], max_tokens=CFG["llm"]["max_tokens"])


def process(path):
    path = pathlib.Path(path)
    text, seconds = transcribe(path)
    if CFG["redaction"]["enabled"]:
        text = redact(text)
    result = {"call_id": hashlib.sha256(path.read_bytes()).hexdigest()[:16], "file": path.name,
              "processed_at": dt.datetime.now().isoformat(timespec="seconds"), "duration_s": round(seconds, 1),
              "transcript": text, "analytics": analyse(text) if text else None}
    out = ROOT / CFG["storage"]["dir"]
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{result['call_id']}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    target = pathlib.Path(sys.argv[1])
    files = sorted(p for p in target.rglob("*") if p.suffix.lower() in AUDIO_EXT) if target.is_dir() else [target]
    for f in files:
        r = process(f)
        a = r["analytics"] or {}
        print(f"{f.name}: {r['duration_s']}s | {a.get('topic')} | {a.get('customer_sentiment')} | escalate={a.get('escalate')}")
        print("   ", (a.get("summary_en") or r["transcript"])[:200])
