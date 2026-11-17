"""pipeline.py: the analysis engine. Cameras -> motion gate -> batched detection -> rules -> tracking -> alerts.

The engine is written so every part can be replaced in tests (detector, cameras, notifier, clock).
"""
import datetime as dt
import logging
import pathlib
import sys
import threading
import time
import uuid

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1] / "common" / "python"))
import cv2  # noqa: E402
import numpy as np  # noqa: E402
from prometheus_client import Counter, Gauge, Histogram  # noqa: E402
from vision.detector import iou  # noqa: E402
from vision.evidence import sha256_bytes  # noqa: E402
from vision.motion import MotionGate  # noqa: E402
from vision.tracker import IoUTracker  # noqa: E402
from vision.video import apply_masks, write_clip  # noqa: E402

log = logging.getLogger("weapons.pipeline")
FRAMES = Counter("cctv_frames_analysed_total", "Frames sent to the detector", ["camera"])
SKIPPED = Counter("cctv_frames_skipped_no_motion_total", "Frames skipped by the motion gate", ["camera"])
ALERTS = Counter("cctv_alerts_total", "Weapon alerts raised", ["camera", "label"])
INFER = Histogram("cctv_inference_seconds", "Detector latency per batch", buckets=(.05, .1, .2, .4, .8, 1.6, 3.2))
CAM_UP = Gauge("cctv_camera_up", "1 when the camera delivered a frame recently", ["camera"])


def weapon_on_person(weapon_box, persons, margin=0.15):
    """True if the weapon box overlaps a person box enlarged by `margin` (arms reach outside the body box)."""
    for p in persons:
        w, h = p[2] - p[0], p[3] - p[1]
        grown = (p[0] - margin * w, p[1] - margin * h, p[2] + margin * w, p[3] + margin * h)
        if iou(weapon_box, grown) > 0 or (grown[0] <= (weapon_box[0] + weapon_box[2]) / 2 <= grown[2]
                                          and grown[1] <= (weapon_box[1] + weapon_box[3]) / 2 <= grown[3]):
            return True
    return False


def draw(rgb, dets, alert_label):
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for d in dets:
        color = (0, 0, 255) if d.label == alert_label else (0, 200, 0)
        x1, y1, x2, y2 = map(int, d.box)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{d.label} {d.score:.2f}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img


class Pipeline:
    def __init__(self, cfg, detector, store, notifier, cameras, media_dir, clock=time.time):
        self.cfg, self.det, self.store, self.notifier, self.clock = cfg, detector, store, notifier, clock
        self.cameras = cameras                                   # {cam_id: CameraStream-like}
        self.cam_cfg = {c["id"]: c for c in cfg["cameras"]}
        self.media = pathlib.Path(media_dir)
        self.media.mkdir(parents=True, exist_ok=True)
        a = cfg["alerting"]
        self.thresholds = {k: float(v) for k, v in a["classes"].items()}
        self.trackers = {c: IoUTracker(min_hits=int(a["min_hits"]), max_misses=2) for c in cameras}
        m = cfg.get("motion", {})
        self.gates = {c: MotionGate(m.get("min_changed_fraction", 0.004), keepalive_s=m.get("keepalive_s", 10))
                      for c in cameras} if m.get("enabled", True) else {}
        self.last_alert = {}                                     # (camera, label) -> ts
        self.pending_clips = []                                  # dicts collecting post-event frames
        self.offline_alerted = set()
        self._stop = threading.Event()

    # ---------------------------------------------------------------- per frame
    def process(self, cam_id, ts, rgb, dets):
        a = self.cfg["alerting"]
        persons = [d.box for d in dets if d.label == a["person_class"]]
        weapons = [d for d in dets if d.label in self.thresholds and d.score >= self.thresholds[d.label]
                   and (not a["require_person_overlap"] or weapon_on_person(d.box, persons))]
        tracker = self.trackers[cam_id]
        tracker.update(weapons, ts)
        new_events = []
        for t in tracker.confirmed():
            key = (cam_id, t.label)
            if t.data.get("alerted") or ts - self.last_alert.get(key, -1e18) < float(a["cooldown_s"]):
                continue
            t.data["alerted"] = True
            self.last_alert[key] = ts
            new_events.append(self._raise(cam_id, ts, rgb, dets, t))
        self._feed_clips(cam_id, ts, rgb)
        return new_events

    def _raise(self, cam_id, ts, rgb, dets, track):
        a = self.cfg["alerting"]
        ev_id = f"{dt.datetime.fromtimestamp(ts):%Y%m%d-%H%M%S}-{cam_id}-{uuid.uuid4().hex[:6]}"
        raw = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()
        annotated = cv2.imencode(".jpg", draw(rgb, dets, track.label), [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
        (self.media / f"{ev_id}_raw.jpg").write_bytes(raw)
        snap = self.media / f"{ev_id}.jpg"
        snap.write_bytes(annotated)
        ev = self.store.add({"id": ev_id, "ts": ts, "camera": cam_id, "label": track.label, "score": round(track.score, 4),
                             "box": list(track.box), "snapshot": str(snap), "sha256": sha256_bytes(raw)})
        ALERTS.labels(cam_id, track.label).inc()
        cam = self.cameras.get(cam_id)
        pre = [f for f in (cam.recent_frames() if cam else []) if f[0] >= ts - float(a["clip_pre_s"])]
        self.pending_clips.append({"id": ev_id, "camera": cam_id, "frames": pre, "until": ts + float(a["clip_post_s"])})
        payload = {"id": ev_id, "type": "weapon_detected", "label": track.label, "score": track.score,
                   "camera": cam_id, "camera_name": self.cam_cfg.get(cam_id, {}).get("name", cam_id),
                   "time": dt.datetime.fromtimestamp(ts).isoformat(timespec="seconds"), "box": list(track.box),
                   "site": self.cfg["site"]["name"], "snapshot_sha256": ev["sha256"]}
        self.notifier.send(payload, str(snap))
        log.warning("ALERT %s %s on %s (%.2f)", ev_id, track.label, cam_id, track.score)
        return ev

    def _feed_clips(self, cam_id, ts, rgb):
        done = []
        for c in self.pending_clips:
            if c["camera"] == cam_id:
                c["frames"].append((ts, rgb))
                if ts >= c["until"]:
                    done.append(c)
        for c in done:
            self.pending_clips.remove(c)
            fps = float(self.cam_cfg.get(cam_id, {}).get("fps", 2))
            threading.Thread(target=self._write_clip, args=(c, fps), daemon=True).start()

    def _write_clip(self, c, fps):
        try:
            path = write_clip(c["frames"], self.media / f"{c['id']}.mp4", fps)
            self.store.set_clip(c["id"], path)
        except Exception as exc:
            log.error("clip %s failed: %s", c["id"], exc)

    # ---------------------------------------------------------------- housekeeping
    def housekeeping(self):
        a, now = self.cfg["alerting"], self.clock()
        for cam_id, cam in self.cameras.items():
            last = cam.status.get("last_frame_ts")
            up = last is not None and now - last < float(a["camera_offline_after_s"])
            CAM_UP.labels(cam_id).set(1 if up else 0)
            if not up and cam_id not in self.offline_alerted and not cam.status.get("finished"):
                self.offline_alerted.add(cam_id)
                self.store.system_alert(cam_id, "camera_offline", cam.status.get("last_error") or "no frames")
                self.notifier.send({"id": f"offline-{cam_id}-{int(now)}", "type": "camera_offline", "label": "camera_offline",
                                    "score": 1.0, "camera": cam_id, "camera_name": self.cam_cfg.get(cam_id, {}).get("name", cam_id),
                                    "time": dt.datetime.fromtimestamp(now).isoformat(timespec="seconds"),
                                    "site": self.cfg["site"]["name"]})
            elif up:
                self.offline_alerted.discard(cam_id)
        for ev in self.store.q("SELECT * FROM events WHERE status='new' AND escalated=0 AND ts<?",
                               (now - float(a["escalate_after_s"]),)):
            self.store.x("UPDATE events SET escalated=1 WHERE id=?", (ev["id"],))
            self.notifier.send({"id": ev["id"], "type": "weapon_detected", "escalation": True, "label": ev["label"],
                                "score": ev["score"], "camera": ev["camera"],
                                "camera_name": self.cam_cfg.get(ev["camera"], {}).get("name", ev["camera"]),
                                "time": dt.datetime.fromtimestamp(ev["ts"]).isoformat(timespec="seconds"),
                                "site": self.cfg["site"]["name"]}, ev["snapshot"], escalation=True)

    # ---------------------------------------------------------------- main loop
    def step(self):
        """Analyse at most one new frame per camera. Returns the number of frames analysed."""
        batch = []
        for cam_id, cam in self.cameras.items():
            item = cam.latest()
            if item is None:
                continue
            ts, rgb = item
            rgb = apply_masks(rgb, self.cam_cfg.get(cam_id, {}).get("privacy_masks"))
            gate = self.gates.get(cam_id)
            if gate and not gate.check(rgb, ts):
                SKIPPED.labels(cam_id).inc()
                self._feed_clips(cam_id, ts, rgb)
                continue
            batch.append((cam_id, ts, rgb))
        bsz = int(self.cfg["detector"]["batch"])
        for i in range(0, len(batch), bsz):
            chunk = batch[i:i + bsz]
            t0 = time.perf_counter()
            results = self.det.detect([c[2] for c in chunk])
            INFER.observe(time.perf_counter() - t0)
            for (cam_id, ts, rgb), dets in zip(chunk, results):
                FRAMES.labels(cam_id).inc()
                self.process(cam_id, ts, rgb, dets)
        return len(batch)

    def run(self):
        last_house = 0.0
        while not self._stop.is_set():
            n = self.step()
            if self.clock() - last_house > 5:
                self.housekeeping()
                last_house = self.clock()
            if n == 0:
                time.sleep(0.02)

    def stop(self):
        self._stop.set()
