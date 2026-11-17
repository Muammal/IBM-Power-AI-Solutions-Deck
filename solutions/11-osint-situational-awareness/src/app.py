"""app.py: the service (scheduler + API + dashboard).

    python src/app.py
API (bearer tokens; viewer < operator(analyst) < admin):
    GET  /health, /metrics
    GET  /api/events?category=&governorate=&min_severity=&hours=24     viewer
    GET  /api/search?q=...                                            viewer (full-text over titles + summaries)
    GET  /api/stories?hours=24                                        viewer
    GET  /api/alerts?hours=24                                         viewer
    GET  /api/sources                                                 viewer (collection health)
    GET  /api/briefings/latest                                        viewer (markdown)
    POST /api/events/{id}/correction                                  operator (analyst fixes -> training data)
"""
import json
import logging
import pathlib
import sys
import threading
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1] / "common" / "python"))
sys.path.insert(0, str(ROOT / "src"))
from power_runtime import load_yaml  # noqa: E402

CFG = load_yaml(ROOT / "config.yaml")

import uvicorn  # noqa: E402
from fastapi import Depends, FastAPI, HTTPException  # noqa: E402
from fastapi.responses import HTMLResponse, PlainTextResponse, Response  # noqa: E402
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402
from vision.auth import TokenAuth  # noqa: E402

from engine import Engine, OsintStore  # noqa: E402
from enrich import CATEGORIES, new_llm  # noqa: E402
from gazetteer import NAMES  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("osint")


def scheduler(engine, cfg):
    last_purge, briefed = 0.0, None
    while True:
        try:
            engine.collect()
            engine.process(limit=100)
            now = time.localtime()
            hhmm = time.strftime("%H:%M", now)
            today = time.strftime("%Y-%m-%d", now)
            if hhmm >= cfg["briefing"]["daily_at"] and briefed != today:
                engine.briefing(today)
                briefed = today
            if time.time() - last_purge > 86400:
                engine.purge()
                last_purge = time.time()
        except Exception as exc:
            log.exception("scheduler error: %s", exc)
        time.sleep(30)


class Correction(BaseModel):
    is_event: bool = True
    category: str | None = None
    governorates: list[str] | None = None
    severity: int | None = Field(default=None, ge=1, le=5)
    note: str = Field(default="", max_length=500)


def create_app(engine, auth):
    store = engine.store
    app = FastAPI(title="Situational awareness (IBM Power)")
    viewer, operator = auth.require("viewer"), auth.require("operator")

    def decode(e):
        e["governorates"] = json.loads(e["governorates"] or "[]")
        e["organizations"] = json.loads(e["organizations"] or "[]")
        e["corrected"] = json.loads(e["corrected"]) if e.get("corrected") else None
        return e

    @app.get("/health")
    def health():
        src = store.q("SELECT id, last_ok, last_error FROM sources")
        failing = [s["id"] for s in src if s["last_error"]]
        return {"status": "ok", "sources": len(src), "failing_sources": failing, "pending_items": len(engine.pending())}

    @app.get("/metrics")
    def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/api/events")
    def events(category: str | None = None, governorate: str | None = None, min_severity: int = 1, hours: int = 24,
               limit: int = 200, user=Depends(viewer)):
        sql, args = "SELECT * FROM events WHERE published>=? AND severity>=?", [time.time() - hours * 3600, min_severity]
        if category:
            sql, args = sql + " AND category=?", args + [category]
        if governorate:
            sql, args = sql + " AND governorates LIKE ?", args + [f'%"{governorate}"%']
        return [decode(e) for e in store.q(sql + " ORDER BY published DESC LIMIT ?", args + [min(limit, 1000)])]

    @app.get("/api/search")
    def search(q: str, user=Depends(viewer)):
        if len(q) < 2:
            raise HTTPException(400, "query too short")
        try:
            return [decode(e) for e in store.search(q)]
        except Exception:
            raise HTTPException(400, "invalid search syntax")

    @app.get("/api/stories")
    def stories(hours: int = 24, user=Depends(viewer)):
        rows = store.q("SELECT id, category, title, first_seen, last_seen, n_items, max_severity FROM stories "
                       "WHERE last_seen>=? ORDER BY max_severity DESC, n_items DESC LIMIT 200", (time.time() - hours * 3600,))
        return rows

    @app.get("/api/alerts")
    def alerts(hours: int = 24, user=Depends(viewer)):
        return store.q("SELECT * FROM alerts WHERE ts>=? ORDER BY ts DESC", (time.time() - hours * 3600,))

    @app.get("/api/sources")
    def sources(user=Depends(viewer)):
        return store.q("SELECT id, last_run, last_ok, items_total, last_error FROM sources")

    @app.get("/api/briefings/latest", response_class=PlainTextResponse)
    def briefing(user=Depends(viewer)):
        r = store.q("SELECT markdown FROM briefings ORDER BY day DESC LIMIT 1")
        return r[0]["markdown"] if r else engine.briefing()

    @app.post("/api/events/{event_id}/correction")
    def correct(event_id: str, body: Correction, user=Depends(operator)):
        if not store.q("SELECT 1 FROM events WHERE id=?", (event_id,)):
            raise HTTPException(404, "not found")
        if body.category and body.category not in CATEGORIES:
            raise HTTPException(400, f"category must be one of {CATEGORIES}")
        if body.governorates and set(body.governorates) - set(NAMES + ["Nationwide", "Outside Iraq"]):
            raise HTTPException(400, "unknown governorate")
        fix = {k: v for k, v in body.model_dump().items() if v is not None}
        fix["by"], fix["at"] = user[0], time.time()
        store.x("UPDATE events SET corrected=? WHERE id=?", (json.dumps(fix, ensure_ascii=False), event_id))
        store.audit(user[0], "event_corrected", event_id, fix)
        return decode(store.q("SELECT * FROM events WHERE id=?", (event_id,))[0])

    @app.get("/", response_class=HTMLResponse)
    def dashboard():
        return (ROOT / "static" / "dashboard.html").read_text(encoding="utf-8")

    return app


if __name__ == "__main__":
    store = OsintStore(ROOT / CFG["storage"]["db"])
    engine = Engine(CFG, store, new_llm(CFG))
    threading.Thread(target=scheduler, args=(engine, CFG), name="scheduler", daemon=True).start()
    app = create_app(engine, TokenAuth(ROOT / CFG["security"]["tokens_file"]))
    uvicorn.run(app, host=CFG["server"]["host"], port=int(CFG["server"]["port"]), log_level="info")
