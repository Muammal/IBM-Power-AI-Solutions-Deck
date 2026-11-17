"""app.py: traffic service: camera readers + pipeline + reviewer API/console.

    python src/app.py
API (bearer tokens, roles):
    GET  /health, /metrics
    GET  /api/stats?camera=&hours=24                 viewer    counts + mean spot speed per 15 min (no plates)
    GET  /api/journeys?section=&hours=24             viewer    travel-time statistics (keyed hashes, no plates)
    GET  /api/violations?status=pending              reviewer
    GET  /api/violations/{id}                        reviewer  incl. signature verification
    GET  /api/violations/{id}/image/{entry|exit}/{plate|overview}
    POST /api/violations/{id}/decision               reviewer  {"decision": "approved"|"rejected", "reason": "..."}
    GET  /api/violations/{id}/evidence.zip           reviewer  approved only: images + manifest + signature + key
There is deliberately NO endpoint to search plates or list reads: reads exist only to match section entry/exit and
expire after retention.reads_hours.
"""
import io
import json
import logging
import pathlib
import statistics
import sys
import threading
import time
import zipfile

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
from vision.evidence import Signer, verify  # noqa: E402

from pipeline import TrafficPipeline  # noqa: E402
from traffic_store import TrafficStore  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("traffic")


def build(cfg=CFG, detector=None, ocr=None, cameras=None, clock_source=None, start=True):
    st = cfg["storage"]
    store = TrafficStore(ROOT / st["db"])
    signer = Signer(ROOT / st["evidence_key"])
    if detector is None:
        from vision.detector import OnnxDetector
        detector = OnnxDetector(ROOT / cfg["detector"]["model"], threads=cfg["detector"]["threads"],
                                min_score=cfg["detector"]["min_score"])
        need = set(cfg["detector"]["vehicle_classes"]) | {cfg["detector"]["plate_class"]}
        if need - set(detector.labels):
            raise SystemExit(f"detector lacks classes {need - set(detector.labels)}: IMPLEMENTATION.md Step 5")
    if ocr is None:
        from ocr import OnnxPlateReader
        ocr = OnnxPlateReader(ROOT / cfg["ocr"]["model"])
    if cameras is None:
        from vision.video import CameraStream
        cameras = {c["id"]: CameraStream(c["id"], c["url"], fps=c.get("fps", 10), buffer_seconds=1)
                   for c in cfg["cameras"]}
    kw = {"clock_source": clock_source} if clock_source else {}
    pipe = TrafficPipeline(cfg, detector, ocr, store, signer, cameras, ROOT / st["media_dir"], **kw)
    if start:
        for cam in cameras.values():
            cam.start()
        threading.Thread(target=pipe.run, name="pipeline", daemon=True).start()
        threading.Thread(target=_retention, args=(store, cfg), name="retention", daemon=True).start()
    return store, pipe, signer


def _retention(store, cfg):
    while True:
        try:
            store.purge(cfg["retention"]["reads_hours"], cfg["retention"]["violations_days"], ROOT / cfg["storage"]["media_dir"])
        except Exception as exc:
            log.error("retention failed: %s", exc)
        time.sleep(600)


class Decision(BaseModel):
    decision: str = Field(pattern="^(approved|rejected)$")
    reason: str = Field(default="", max_length=500)


def create_app(store, pipe, signer, auth):
    app = FastAPI(title="Traffic ANPR and section speed (IBM Power)")
    viewer, reviewer = auth.require("viewer"), auth.require("reviewer")

    def violation(vid):
        rows = store.q("SELECT * FROM violations WHERE id=?", (vid,))
        if not rows:
            raise HTTPException(404, "not found")
        v = rows[0]
        v["manifest"] = json.loads(v["manifest"])
        return v

    def read(rid):
        r = store.q("SELECT * FROM reads WHERE id=?", (rid,))
        if not r:
            raise HTTPException(410, "read expired")
        return r[0]

    @app.get("/health")
    def health():
        up = sum(1 for c in pipe.cameras.values() if c.status.get("last_frame_ts") and time.time() - c.status["last_frame_ts"] < 30)
        return {"status": "ok", "cameras_up": up, "cameras_total": len(pipe.cameras), "clock_offset_s": pipe.clock_offset}

    @app.get("/metrics")
    def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/api/stats")
    def stats(camera: str | None = None, hours: int = 24, user=Depends(viewer)):
        since = time.time() - hours * 3600
        sql, args = "SELECT * FROM stats WHERE bin>=?", [since]
        if camera:
            sql, args = sql + " AND camera=?", args + [camera]
        rows = store.q(sql + " ORDER BY bin", args)
        for r in rows:
            r["mean_spot_kmh"] = round(r["speed_sum"] / r["speed_n"], 1) if r["speed_n"] else None
        return rows

    @app.get("/api/journeys")
    def journeys(section: str, hours: int = 24, user=Depends(viewer)):
        t = [r["travel_s"] for r in store.q("SELECT travel_s FROM journeys WHERE section=? AND entry_ts>=?",
                                            (section, time.time() - hours * 3600))]
        if not t:
            return {"section": section, "count": 0}
        return {"section": section, "count": len(t), "median_s": statistics.median(t),
                "p85_s": sorted(t)[int(0.85 * (len(t) - 1))]}

    @app.get("/api/violations")
    def violations(status: str = "pending", limit: int = 100, user=Depends(reviewer)):
        rows = store.q("SELECT id, created, section, plate, measured_kmh, lower_bound_kmh, limit_kmh, status, reviewer "
                       "FROM violations WHERE status=? ORDER BY created DESC LIMIT ?", (status, min(limit, 500)))
        return rows

    @app.get("/api/violations/{vid}")
    def get_violation(vid: str, user=Depends(reviewer)):
        v = violation(vid)
        v["signature_valid"] = verify(v["manifest"], v["signature"], signer.public_pem())
        store.audit(user[0], "view_violation", vid)
        return v

    @app.get("/api/violations/{vid}/image/{which}/{kind}")
    def image(vid: str, which: str, kind: str, user=Depends(reviewer)):
        if which not in ("entry", "exit") or kind not in ("plate", "overview"):
            raise HTTPException(400, "which=entry|exit, kind=plate|overview")
        v = violation(vid)
        r = read(v[f"{which}_read"])
        path = r["crop" if kind == "plate" else "overview"]
        if not path or not pathlib.Path(path).exists():
            raise HTTPException(410, "image expired")
        return FileResponse(path, media_type="image/jpeg")

    @app.post("/api/violations/{vid}/decision")
    def decide(vid: str, body: Decision, user=Depends(reviewer)):
        try:
            store.decide(vid, body.decision, user[0], body.reason)
        except KeyError:
            raise HTTPException(404, "not found")
        except PermissionError as exc:
            raise HTTPException(409, str(exc))
        return get_violation(vid, user)

    @app.get("/api/violations/{vid}/evidence.zip")
    def evidence(vid: str, user=Depends(reviewer)):
        v = violation(vid)
        if v["status"] != "approved":
            raise HTTPException(409, "evidence packages are issued for approved violations only")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("manifest.json", json.dumps(v["manifest"], ensure_ascii=False, indent=2, sort_keys=True))
            z.writestr("signature.ed25519.hex", v["signature"])
            z.writestr("public_key.pem", signer.public_pem())
            z.writestr("review.json", json.dumps({k: v[k] for k in ("status", "reviewer", "reviewed_at", "reason")}))
            for which in ("entry", "exit"):
                r = read(v[f"{which}_read"])
                for kind in ("crop", "overview"):
                    if r[kind] and pathlib.Path(r[kind]).exists():
                        z.write(r[kind], f"{which}_{kind}.jpg")
        store.audit(user[0], "evidence_exported", vid)
        return Response(buf.getvalue(), media_type="application/zip",
                        headers={"Content-Disposition": f'attachment; filename="{vid}.zip"'})

    @app.get("/", response_class=HTMLResponse)
    def console():
        return (ROOT / "static" / "review.html").read_text(encoding="utf-8")

    return app


if __name__ == "__main__":
    store, pipe, signer = build()
    app = create_app(store, pipe, signer, TokenAuth(ROOT / CFG["security"]["tokens_file"]))
    uvicorn.run(app, host=CFG["server"]["host"], port=int(CFG["server"]["port"]), log_level="info")
