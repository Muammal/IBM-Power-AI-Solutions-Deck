"""pipeline.py: camera frames -> vehicles + plates -> tracks -> line crossing -> OCR vote -> read -> section speed.

Per camera:
  1. detect vehicles and plates (RT-DETRv2 fine-tuned, ONNX Runtime)
  2. track vehicles (IoU tracker); attach the best plate box inside each vehicle box to the track
  3. when a vehicle's bottom-centre crosses a lane's trigger line: interpolate the crossing time
  4. when the track ends: OCR the best plate crops, vote, normalise, validate -> PlateRead
  5. at a section's exit camera: find the entry read -> average speed -> violation (signed evidence) if over
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
from prometheus_client import Counter, Gauge  # noqa: E402
from vision.evidence import sha256_bytes  # noqa: E402
from vision.tracker import IoUTracker  # noqa: E402

from plates import is_ambiguous, normalise, plate_key, repair, vote  # noqa: E402
from speed import Homography, chrony_offset, crossing_time, section_speed  # noqa: E402

log = logging.getLogger("traffic.pipeline")
READS = Counter("traffic_plate_reads_total", "Vehicles that crossed a trigger line", ["camera", "result"])
VIOLATIONS = Counter("traffic_violations_total", "Section-speed violations created", ["section"])
CLOCK = Gauge("traffic_clock_offset_seconds", "chrony offset from NTP")
CAM_UP = Gauge("traffic_camera_up", "1 when the camera delivered a frame recently", ["camera"])


def utc(ts):
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def sharpness(rgb):
    return float(cv2.Laplacian(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())


class TrafficPipeline:
    def __init__(self, cfg, detector, ocr, store, signer, cameras, media_dir, clock_source=chrony_offset):
        self.cfg, self.det, self.ocr, self.store, self.signer = cfg, detector, ocr, store, signer
        self.cameras = cameras
        self.cam_cfg = {c["id"]: c for c in cfg["cameras"]}
        self.media = pathlib.Path(media_dir)
        self.media.mkdir(parents=True, exist_ok=True)
        self.trackers = {c: IoUTracker(iou_threshold=0.25, min_hits=2, max_misses=3) for c in cameras}
        self.homography = {c["id"]: Homography(c["ground_points"]["image"], c["ground_points"]["world"])
                           for c in cfg["cameras"] if c.get("ground_points")}
        self.sections_by_exit = {}
        for s in cfg.get("sections", []):
            self.sections_by_exit.setdefault(s["exit"], []).append(s)
        self.clock_source = clock_source
        self.clock_offset, self.clock_checked = None, 0.0
        self._stop = threading.Event()

    # ------------------------------------------------------------ timing
    def current_offset(self):
        now = time.time()
        if now - self.clock_checked > float(self.cfg["timing"]["clock_check_interval_s"]):
            self.clock_offset, self.clock_checked = self.clock_source(), now
            if self.clock_offset is not None:
                CLOCK.set(self.clock_offset)
        return self.clock_offset

    # ------------------------------------------------------------ per frame
    def process(self, cam_id, ts, rgb, dets):
        d = self.cfg["detector"]
        vehicles = [x for x in dets if x.label in d["vehicle_classes"]]
        plates = [x for x in dets if x.label == d["plate_class"]]
        tracks, ended = self.trackers[cam_id].update(vehicles, ts)
        for t in tracks:
            if t.misses:
                continue
            self._attach_plate(t, plates, rgb)
            self._check_crossing(cam_id, t, rgb)
        reads = []
        for t in ended:
            r = self._finish(cam_id, t)
            if r:
                reads.append(r)
        return reads

    def _attach_plate(self, track, plates, rgb):
        vx1, vy1, vx2, vy2 = track.box
        inside = [p for p in plates if vx1 <= (p.box[0] + p.box[2]) / 2 <= vx2 and vy1 <= (p.box[1] + p.box[3]) / 2 <= vy2]
        if not inside:
            return
        p = max(inside, key=lambda p: p.score)
        x1, y1, x2, y2 = (int(v) for v in p.box)
        crop = rgb[max(0, y1 - 2):y2 + 2, max(0, x1 - 2):x2 + 2]
        if crop.size == 0 or crop.shape[1] < 30:                     # too small to read
            return
        quality = p.score * min(1.0, sharpness(crop) / 300.0) * min(1.0, crop.shape[1] / 120.0)
        crops = track.data.setdefault("crops", [])
        crops.append((quality, crop.copy()))
        crops.sort(key=lambda c: -c[0])
        del crops[int(self.cfg["ocr"]["crops_per_vehicle"]):]

    def _check_crossing(self, cam_id, track, rgb):
        if "crossed" in track.data or len(track.history) < 2:
            return
        (t0, b0, _), (t1, b1, _) = track.history[-2], track.history[-1]
        p0, p1 = ((b0[0] + b0[2]) / 2, b0[3]), ((b1[0] + b1[2]) / 2, b1[3])
        for lane in self.cam_cfg[cam_id].get("lanes", []):
            tc = crossing_time(p0, t0, p1, t1, lane["line"])
            if tc is not None:
                track.data.update(crossed=tc, lane=lane["id"], overview=rgb.copy(), frame_dt=t1 - t0,
                                  clock_offset=self.current_offset())
                return

    def _finish(self, cam_id, track):
        if "crossed" not in track.data:
            return None
        o = self.cfg["ocr"]
        texts = []
        for _, crop in track.data.get("crops", []):
            raw, conf = self.ocr.read(crop)
            texts.append((repair(normalise(raw), o["formats"]), conf))
        plate, conf, agreement = vote(texts)
        if is_ambiguous(plate, o["formats"]):
            conf = min(conf, float(o["min_confidence"]) * 0.99)  # keep for statistics, never for enforcement
        if not plate:
            READS.labels(cam_id, "unread").inc()
        rid = f"{dt.datetime.fromtimestamp(track.data['crossed']):%Y%m%d-%H%M%S%f}-{cam_id}-{uuid.uuid4().hex[:4]}"
        crop_path = overview_path = crop_sha = ov_sha = None
        if plate:
            crop_bytes = cv2.imencode(".jpg", cv2.cvtColor(track.data["crops"][0][1], cv2.COLOR_RGB2BGR))[1].tobytes()
            ov_bytes = cv2.imencode(".jpg", cv2.cvtColor(track.data["overview"], cv2.COLOR_RGB2BGR),
                                    [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()
            crop_path, overview_path = self.media / f"{rid}_plate.jpg", self.media / f"{rid}_overview.jpg"
            crop_path.write_bytes(crop_bytes)
            overview_path.write_bytes(ov_bytes)
            crop_sha, ov_sha = sha256_bytes(crop_bytes), sha256_bytes(ov_bytes)
            READS.labels(cam_id, "read" if conf >= o["min_confidence"] else "low_confidence").inc()
        hom = self.homography.get(cam_id)
        spot = hom.speed_kmh([(t, ((b[0] + b[2]) / 2, b[3])) for t, b, _ in track.history]) if hom else None
        read = {"id": rid, "ts": track.data["crossed"], "camera": cam_id, "lane": track.data["lane"], "plate": plate,
                "plate_key": plate_key(plate, self.cfg["retention"]["hmac_key"]) if plate else None,
                "confidence": round(conf, 4), "agreement": round(agreement, 3), "vehicle": track.label,
                "crop": str(crop_path) if crop_path else None, "overview": str(overview_path) if overview_path else None,
                "crop_sha256": crop_sha, "overview_sha256": ov_sha, "frame_dt": track.data["frame_dt"],
                "clock_offset": track.data["clock_offset"], "spot_kmh": round(spot, 1) if spot else None}
        self.store.add_read(read)
        if plate and conf >= o["min_confidence"]:
            self._sections(read)
        return read

    def _sections(self, exit_read):
        for s in self.sections_by_exit.get(exit_read["camera"], []):
            entry = self.store.find_entry(exit_read["plate"], s["entry"], exit_read["ts"], float(s["max_travel_s"]))
            if not entry or entry["confidence"] < self.cfg["ocr"]["min_confidence"]:
                continue
            self.store.add_journey(s["id"], exit_read["plate_key"], entry["ts"], exit_read["ts"] - entry["ts"])
            measured, lower, unc = section_speed(float(s["distance_m"]), entry["ts"], exit_read["ts"],
                                                 entry["frame_dt"], exit_read["frame_dt"],
                                                 (entry["clock_offset"] or 0.0, exit_read["clock_offset"] or 0.0))
            if lower <= float(s["limit_kmh"]) + float(s["tolerance_kmh"]):
                continue
            max_off = float(self.cfg["timing"]["max_clock_offset_s"])
            offs = [entry["clock_offset"], exit_read["clock_offset"]]
            if any(o is None or abs(o) > max_off for o in offs):
                log.error("violation suppressed for %s: clock offsets %s exceed %.3fs or unknown", s["id"], offs, max_off)
                continue
            self._violation(s, entry, exit_read, measured, lower, unc)

    def _violation(self, s, entry, exit_read, measured, lower, unc):
        vid = f"V-{dt.datetime.fromtimestamp(exit_read['ts']):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
        manifest = {
            "violation_id": vid, "authority": self.cfg["site"]["authority"], "type": "section_average_speed",
            "section": s["id"], "plate": exit_read["plate"], "distance_m": s["distance_m"],
            "limit_kmh": s["limit_kmh"], "tolerance_kmh": s["tolerance_kmh"],
            "measured_kmh": round(measured, 2), "lower_bound_kmh": round(lower, 2), "timing_uncertainty_s": round(unc, 4),
            "entry": {k: entry[k] for k in ("id", "ts", "camera", "lane", "confidence", "crop_sha256", "overview_sha256",
                                            "clock_offset")},
            "exit": {k: exit_read[k] for k in ("id", "ts", "camera", "lane", "confidence", "crop_sha256",
                                                "overview_sha256", "clock_offset")},
            "entry_time_utc": utc(entry["ts"]),
            "exit_time_utc": utc(exit_read["ts"]),
        }
        self.store.add_violation({"id": vid, "section": s["id"], "plate": exit_read["plate"], "entry_read": entry["id"],
                                  "exit_read": exit_read["id"], "measured_kmh": round(measured, 2),
                                  "lower_bound_kmh": round(lower, 2), "limit_kmh": s["limit_kmh"],
                                  "tolerance_kmh": s["tolerance_kmh"], "manifest": manifest,
                                  "signature": self.signer.sign(manifest)})
        VIOLATIONS.labels(s["id"]).inc()
        log.warning("violation %s %s %.1f km/h (lower bound %.1f) on %s", vid, exit_read["plate"], measured, lower, s["id"])

    # ------------------------------------------------------------ loop
    def step(self):
        n = 0
        for cam_id, cam in self.cameras.items():
            item = cam.latest()
            if item is None:
                continue
            ts, rgb = item
            dets = self.det.detect([rgb])[0]
            self.process(cam_id, ts, rgb, dets)
            n += 1
        return n

    def run(self):
        last = 0.0
        while not self._stop.is_set():
            if self.step() == 0:
                time.sleep(0.005)
            if time.time() - last > 10:
                last = time.time()
                for cam_id, cam in self.cameras.items():
                    lf = cam.status.get("last_frame_ts")
                    CAM_UP.labels(cam_id).set(1 if lf and time.time() - lf < 30 else 0)

    def stop(self):
        self._stop.set()


def blank_frame(h=720, w=1280):
    return np.zeros((h, w, 3), np.uint8)
