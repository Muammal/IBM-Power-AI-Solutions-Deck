"""app.py: the service. Starts camera readers + the analysis engine, and serves the operator API and console.

    python src/app.py                      # production entry point (systemd: scripts/serve.sh)
API (bearer token, roles from tokens.yaml):
    GET  /health                           liveness + camera status (no auth, bind to localhost / LB health check)
    GET  /metrics                          Prometheus metrics (no auth, scrape from the monitoring network only)
    GET  /api/cameras                      viewer
    GET  /api/events?status=new&limit=50   viewer
    GET  /api/events/{id}/snapshot|clip    viewer
    POST /api/events/{id}/review           operator   {"verdict": "confirmed"|"false_alarm", "note": "..."}
    GET  /api/integrity                    admin      hash-chain verification
    GET  /                                 operator console (HTML)
"""
import logging
import pathlib
import sys
import threading
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1] / "common" / "python"))
sys.path.insert(0, str(ROOT / "src"))
from power_runtime import configure_threads, load_yaml  # noqa: E402

CFG = load_yaml(ROOT / "config.yaml")
configure_threads(CFG["detector"]["threads"])

import uvicorn  # noqa: E402
from fastapi import Depends, FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse, HTMLResponse, Response  # noqa: E402
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402
from vision.auth import TokenAuth  # noqa: E402

from events import EventStore  # noqa: E402
from notify import Notifier  # noqa: E402
from pipeline import Pipeline  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("weapons")


def build(cfg=CFG, detector=None, cameras=None, notifier=None, start=True):
    """Wire everything together. Tests pass fakes for detector/cameras/notifier."""
    st = cfg["storage"]
    store = EventStore(ROOT / st["db"])
    if detector is None:
        from vision.detector import OnnxDetector
        detector = OnnxDetector(ROOT / cfg["detector"]["model"], threads=cfg["detector"]["threads"],
                                min_score=cfg["detector"]["min_score"])
        missing = set(cfg["alerting"]["classes"]) - set(detector.labels)
        if missing:
            raise SystemExit(f"model has no classes {missing}: train/export the weapon model first (IMPLEMENTATION.md)")
    if cameras is None:
        from vision.video import CameraStream
        buf = float(cfg["alerting"]["clip_pre_s"]) + 1
        cameras = {c["id"]: CameraStream(c["id"], c["url"], fps=c.get("fps", 2), buffer_seconds=buf)
                   for c in cfg["cameras"] if c.get("enabled", True)}
    notifier = notifier or Notifier(cfg["notify"], cfg["site"]["name"])
    pipe = Pipeline(cfg, detector, store, notifier, cameras, ROOT / st["media_dir"])
    if start:
        for cam in cameras.values():
            if hasattr(cam, "start") and not cam.is_alive():
                cam.start()
        threading.Thread(target=pipe.run, name="pipeline", daemon=True).start()
        threading.Thread(target=_retention_loop, args=(store, cfg), name="retention", daemon=True).start()
    return store, pipe


def _retention_loop(store, cfg):
    st = cfg["storage"]
    while True:
        try:
            neg = ROOT / "data" / "training" / "negatives" if st.get("keep_false_alarms_for_training") else None
            n = store.purge(st["retention_days"], st["confirmed_retention_days"], ROOT / st["media_dir"], neg)
            if n:
                log.info("retention: purged %d events", n)
        except Exception as exc:
            log.error("retention failed: %s", exc)
        time.sleep(3600)


class Review(BaseModel):
    verdict: str = Field(pattern="^(confirmed|false_alarm)$")
    note: str = Field(default="", max_length=1000)


def create_app(store, pipe, auth):
    app = FastAPI(title="CCTV weapon detection (IBM Power)")
    viewer, operator, admin = auth.require("viewer"), auth.require("operator"), auth.require("admin")

    def media(event_id, kind):
        ev = store.get(event_id)
        path = ev and ev.get(kind)
        if not path or not pathlib.Path(path).exists():
            raise HTTPException(404, f"no {kind} for this event")
        return path

    @app.get("/health")
    def health():
        cams = {c: {k: s.status.get(k) for k in ("connected", "last_frame_ts", "reconnects", "last_error")}
                for c, s in pipe.cameras.items()}
        up = sum(1 for c in cams.values() if c["last_frame_ts"] and time.time() - c["last_frame_ts"] < 60)
        return {"status": "ok", "cameras_up": up, "cameras_total": len(cams)}

    @app.get("/metrics")
    def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/api/cameras")
    def cameras(user=Depends(viewer)):
        return [dict(s.status, name=pipe.cam_cfg.get(c, {}).get("name", c)) for c, s in pipe.cameras.items()]

    @app.get("/api/events")
    def events(status: str | None = None, limit: int = 50, user=Depends(viewer)):
        return store.list(status, min(limit, 500))

    @app.get("/api/events/{event_id}")
    def event(event_id: str, user=Depends(viewer)):
        ev = store.get(event_id)
        if not ev:
            raise HTTPException(404, "not found")
        return ev

    @app.get("/api/events/{event_id}/snapshot")
    def snapshot(event_id: str, user=Depends(viewer)):
        store.audit(user[0], "view_snapshot", event_id)
        return FileResponse(media(event_id, "snapshot"), media_type="image/jpeg")

    @app.get("/api/events/{event_id}/clip")
    def clip(event_id: str, user=Depends(viewer)):
        store.audit(user[0], "view_clip", event_id)
        return FileResponse(media(event_id, "clip"), media_type="video/mp4")

    @app.post("/api/events/{event_id}/review")
    def review(event_id: str, body: Review, user=Depends(operator)):
        try:
            store.review(event_id, body.verdict, user[0], body.note)
        except KeyError:
            raise HTTPException(404, "not found")
        return store.get(event_id)

    @app.get("/api/integrity")
    def integrity(user=Depends(admin)):
        broken = store.verify_chain()
        return {"intact": broken is None, "first_problem": broken}

    @app.get("/", response_class=HTMLResponse)
    def console():
        return (ROOT / "static" / "console.html").read_text(encoding="utf-8")

    return app


if __name__ == "__main__":
    store, pipe = build()
    app = create_app(store, pipe, TokenAuth(ROOT / CFG["security"]["tokens_file"]))
    uvicorn.run(app, host=CFG["server"]["host"], port=int(CFG["server"]["port"]), log_level="info")
