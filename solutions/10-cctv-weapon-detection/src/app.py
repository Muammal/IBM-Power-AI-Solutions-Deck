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


def effective_config(cfg):
    """Zero-shot mode replaces thresholds, confirmations and analysis fps with its own tested values."""
    if cfg["detector"].get("backend") != "zeroshot":
        return cfg
    import copy
    cfg = copy.deepcopy(cfg)
    zs = cfg["detector"]["zeroshot"]
    cfg["alerting"]["classes"] = dict(zs["alert_thresholds"])
    cfg["alerting"]["min_hits"] = int(zs["min_hits"])
    cfg["alerting"]["confirm"] = "window"         # 2 s between analysed frames: box tracking would break
    cfg["alerting"]["window_s"] = float(zs.get("window_s", 6))
    for cam in cfg["cameras"]:
        cam["fps"] = float(zs["fps"])
    return cfg


def make_detector(cfg):
    d = cfg["detector"]
    if d.get("backend") == "zeroshot":
        from vision.zeroshot import ZeroShotDetector
        zs = d["zeroshot"]
        log.warning("ZERO-SHOT detector (%s): interim mode with more false alarms; train a site model (Steps 5-7)",
                    zs["model"])
        return ZeroShotDetector(zs["model"], zs["queries"], zs["input_size"], d["threads"], zs["min_score"])
    from vision.detector import OnnxDetector
    det = OnnxDetector(ROOT / d["model"], threads=d["threads"], min_score=d["min_score"])
    missing = set(cfg["alerting"]["classes"]) - set(det.labels)
    if missing:
        raise SystemExit(f"model has no classes {missing}: train/export the weapon model first (IMPLEMENTATION.md), "
                         "or start in zero-shot mode: DETECTOR_BACKEND=zeroshot")
    return det


def build(cfg=CFG, detector=None, cameras=None, notifier=None, start=True):
    """Wire everything together. Tests pass fakes for detector/cameras/notifier."""
    cfg = effective_config(cfg)
    st = cfg["storage"]
    store = EventStore(ROOT / st["db"])
    if detector is None:
        detector = make_detector(cfg)
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
        path = media(event_id, "clip")                     # WebM (VP9) on Power, MP4 (H.264) where libx264 exists
        return FileResponse(path, media_type="video/webm" if path.endswith(".webm") else "video/mp4")

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
