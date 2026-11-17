"""Tests for the traffic solution: plate text, timing maths, the full read -> section speed -> violation path,
clock-safety, retention, and the reviewer API (decision workflow, evidence package, roles).

    cd solutions/12-traffic-anpr-speed && python -m pytest -q tests
"""
import copy
import hashlib
import io
import json
import pathlib
import sys
import time
import zipfile

import numpy as np
import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from app import CFG, build, create_app  # noqa: E402
from plates import is_ambiguous, normalise, plate_key, repair, vote  # noqa: E402
from speed import Homography, crossing_time, parse_chrony, section_speed  # noqa: E402
from vision.auth import TokenAuth  # noqa: E402
from vision.detector import Detection  # noqa: E402
from vision.evidence import verify  # noqa: E402

FORMATS = CFG["ocr"]["formats"]


# ---------------------------------------------------------------- unit tests
def test_plate_text():
    assert normalise("١٢٣ ٤٥٦ ب") == "123456"                 # Arabic-Indic digits, Arabic letter dropped
    assert normalise(" 12-345 a ") == "12345A"
    assert repair("12I456", FORMATS) == "121456"                   # invalid as read, valid after digit fix
    assert repair("I23456", FORMATS) == "I23456"                   # valid as read (letter + digits format) ...
    assert is_ambiguous("I23456", FORMATS)                         # ... but also valid as 123456: not usable for fines
    assert not is_ambiguous("123456", FORMATS)
    assert repair("!!!", FORMATS) is None
    assert vote([("123456", 0.9), ("123456", 0.8), ("128456", 0.95)])[0] == "123456"
    assert plate_key("123456", "k1") != plate_key("123456", "k2")


def test_crossing_and_speed():
    line = [[0, 100], [200, 100]]
    assert crossing_time((50, 90), 10.0, (50, 110), 10.1, line) == pytest.approx(10.05)
    assert crossing_time((50, 90), 10.0, (50, 95), 10.1, line) is None          # did not cross
    assert crossing_time((500, 90), 10.0, (500, 110), 10.1, line) is None       # other lane
    measured, lower, unc = section_speed(5000, 0.0, 150.0, 0.1, 0.1, (0.01, -0.02))
    assert measured == pytest.approx(120.0)
    assert lower < measured and unc == pytest.approx(0.13)
    assert parse_chrony("System time     : 0.000123 seconds slow of NTP time") == pytest.approx(-0.000123)


def test_homography_spot_speed():
    h = Homography([[0, 0], [100, 0], [100, 100], [0, 100]], [[0, 0], [10, 0], [10, 10], [0, 10]])  # 10 px = 1 m
    pts = [(t * 0.1, (50, 100 - t * 25)) for t in range(8)]       # 2.5 m per 0.1 s = 25 m/s = 90 km/h
    assert h.speed_kmh(pts) == pytest.approx(90.0, rel=1e-3)


# ---------------------------------------------------------------- pipeline
class FakeOCR:
    def __init__(self, text, conf=0.97):
        self.text, self.conf = text, conf

    def read(self, crop):
        return self.text, self.conf


class ScriptedCamera:
    """A car driving down the image (crossing y=600) at a given start time."""
    def __init__(self, t0, frames=14, dt=0.1):
        self.items = [(t0 + i * dt, i) for i in range(frames)]
        self.status = {"last_frame_ts": time.time(), "connected": True}

    def latest(self):
        if not self.items:
            return None
        ts, i = self.items.pop(0)
        img = np.random.default_rng(i).integers(0, 255, (720, 1280, 3), dtype=np.uint8)
        return ts, img


class CarDetector:
    """Vehicle box moving 50 px/frame; plate box inside it. After 10 frames the car leaves (track ends)."""
    def __init__(self):
        self.i = {}

    def detect(self, frames):
        out = []
        for _ in frames:
            k = self.i.get("n", 0)
            self.i["n"] = k + 1
            frame_no = k % 14
            if frame_no >= 10:
                out.append([])
                continue
            y = 300 + frame_no * 50
            out.append([Detection("car", 0.9, (300, y - 200, 500, y)),
                         Detection("license_plate", 0.8, (360, y - 60, 440, y - 30))])
        return out


@pytest.fixture
def cfg(tmp_path):
    c = copy.deepcopy(CFG)
    c["storage"].update(db=str(tmp_path / "t.db"), media_dir=str(tmp_path / "media"),
                        evidence_key=str(tmp_path / "keys" / "k.pem"))
    c["timing"]["clock_check_interval_s"] = 0
    return c


def drive(cfg, entry_t0, exit_t0, plate="123456", clock=lambda: 0.002):
    cams = {"hwy1-km12-n": ScriptedCamera(entry_t0), "hwy1-km17-n": ScriptedCamera(exit_t0)}
    store, pipe, signer = build(cfg, detector=CarDetector(), ocr=FakeOCR(plate), cameras=cams,
                                clock_source=clock, start=False)
    for _ in range(40):
        if not pipe.step():
            break
    return store, pipe, signer


def test_fast_car_creates_signed_violation(cfg):
    now = time.time() - 600
    store, _, signer = drive(cfg, now, now + 120.0)          # 5 km in 120 s = 150 km/h > 110 + 10
    reads = store.q("SELECT * FROM reads ORDER BY ts")
    assert [r["camera"] for r in reads] == ["hwy1-km12-n", "hwy1-km17-n"] and reads[0]["plate"] == "123456"
    v = store.q("SELECT * FROM violations")
    assert len(v) == 1 and v[0]["status"] == "pending"
    assert v[0]["measured_kmh"] == pytest.approx(150.0, abs=0.5)
    assert v[0]["lower_bound_kmh"] < v[0]["measured_kmh"]
    assert verify(json.loads(v[0]["manifest"]), v[0]["signature"], signer.public_pem())
    m = json.loads(v[0]["manifest"])
    m["measured_kmh"] = 99
    assert not verify(m, v[0]["signature"], signer.public_pem())           # any edit breaks the signature


def test_legal_speed_and_tolerance(cfg):
    now = time.time() - 600
    store, _, _ = drive(cfg, now, now + 155.0)               # 116 km/h: over the limit but inside the tolerance
    assert store.q("SELECT * FROM violations") == []
    assert len(store.q("SELECT * FROM journeys")) == 1       # still counted in anonymous travel-time statistics


def test_clock_problems_block_violations(cfg):
    now = time.time() - 600
    store, _, _ = drive(cfg, now, now + 100.0, clock=lambda: None)          # chrony unreadable
    assert store.q("SELECT * FROM violations") == []
    store, _, _ = drive(cfg, now + 1000, now + 1100.0, clock=lambda: 0.5)   # 500 ms off
    assert store.q("SELECT * FROM violations") == []


def test_ambiguous_plate_never_fines(cfg):
    now = time.time() - 600
    cams = {"hwy1-km12-n": ScriptedCamera(now), "hwy1-km17-n": ScriptedCamera(now + 100)}
    store, pipe, _ = build(cfg, detector=CarDetector(), ocr=FakeOCR("I23456"), cameras=cams,
                           clock_source=lambda: 0.001, start=False)
    while pipe.step():
        pass
    assert len(store.q("SELECT * FROM reads")) == 2 and store.q("SELECT * FROM violations") == []


def test_low_confidence_reads_never_fine(cfg):
    now = time.time() - 600
    cams = {"hwy1-km12-n": ScriptedCamera(now), "hwy1-km17-n": ScriptedCamera(now + 100)}
    store, pipe, _ = build(cfg, detector=CarDetector(), ocr=FakeOCR("123456", conf=0.5), cameras=cams,
                           clock_source=lambda: 0.001, start=False)
    while pipe.step():
        pass
    assert len(store.q("SELECT * FROM reads")) == 2 and store.q("SELECT * FROM violations") == []


def test_retention(cfg):
    now = time.time() - 2 * 86400                              # two days ago
    store, _, _ = drive(cfg, now, now + 100.0)                 # violation (keeps its 2 reads)
    drive(cfg, now + 500, now + 700.0, plate="777777")         # 90 km/h, no violation (same db)
    n_reads, n_v = store.purge(24, 365, cfg["storage"]["media_dir"])
    assert n_reads == 2 and n_v == 0
    left = store.q("SELECT plate FROM reads")
    assert {r["plate"] for r in left} == {"123456"}
    assert store.q("SELECT count(*) n FROM journeys")[0]["n"] == 2   # anonymous statistics survive


def test_reviewer_api(cfg, tmp_path):
    from fastapi.testclient import TestClient
    now = time.time() - 600
    store, pipe, signer = drive(cfg, now, now + 120.0)
    (tmp_path / "tokens.yaml").write_text(yaml.safe_dump({"users": [
        {"name": "officer.k", "role": "reviewer", "sha256": hashlib.sha256(b"rev").hexdigest()},
        {"name": "analyst", "role": "viewer", "sha256": hashlib.sha256(b"view").hexdigest()}]}))
    c = TestClient(create_app(store, pipe, signer, TokenAuth(tmp_path / "tokens.yaml")))
    R, V = {"Authorization": "Bearer rev"}, {"Authorization": "Bearer view"}
    assert c.get("/api/violations", headers=V).status_code == 403          # viewers never see plates
    vid = c.get("/api/violations", headers=R).json()[0]["id"]
    assert c.get(f"/api/violations/{vid}", headers=R).json()["signature_valid"] is True
    assert c.get(f"/api/violations/{vid}/image/exit/plate", headers=R).status_code == 200
    assert c.get(f"/api/violations/{vid}/evidence.zip", headers=R).status_code == 409   # not approved yet
    assert c.post(f"/api/violations/{vid}/decision", json={"decision": "approved"}, headers=R).status_code == 200
    assert c.post(f"/api/violations/{vid}/decision", json={"decision": "rejected"}, headers=R).status_code == 409
    z = zipfile.ZipFile(io.BytesIO(c.get(f"/api/violations/{vid}/evidence.zip", headers=R).content))
    assert {"manifest.json", "signature.ed25519.hex", "public_key.pem", "entry_crop.jpg", "exit_overview.jpg"} <= set(z.namelist())
    manifest = json.loads(z.read("manifest.json"))
    assert verify(manifest, z.read("signature.ed25519.hex").decode(), z.read("public_key.pem").decode())
    assert c.get("/api/stats", headers=V).status_code == 200
    assert c.get("/api/journeys?section=hwy1-n-km12-17", headers=V).json()["count"] == 1
    paths = {r.path for r in c.app.routes}
    assert not any("reads" in p or "plate" in p.split("/")[-1] for p in paths if p.startswith("/api"))  # no plate search
