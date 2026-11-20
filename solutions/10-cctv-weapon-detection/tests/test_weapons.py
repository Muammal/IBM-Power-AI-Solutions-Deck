"""Tests for the weapon-detection service: alert rules, tracking, evidence chain, retention, API auth and review.

    cd solutions/10-cctv-weapon-detection && python -m pytest -q tests
They use a scripted fake detector and fake cameras, so they run anywhere without a model or camera.
"""
import copy
import hashlib
import pathlib
import sys
import time

import numpy as np
import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from app import CFG, build, create_app  # noqa: E402
from vision.auth import TokenAuth  # noqa: E402
from vision.detector import Detection  # noqa: E402

PERSON = Detection("person", 0.9, (100, 100, 200, 400))
GUN = Detection("handgun", 0.8, (180, 220, 220, 260))      # in the person's hand
GUN_FAR = Detection("handgun", 0.8, (500, 50, 540, 90))     # nobody near it (e.g. poster on a wall)
PHONE_LIKE = Detection("handgun", 0.4, (180, 220, 220, 260))  # below threshold


class FakeDetector:
    labels = ["person", "handgun", "rifle", "knife"]

    def __init__(self, script):
        self.script = list(script)

    def detect(self, frames):
        return [self.script.pop(0) if self.script else [] for _ in frames]


class FakeCamera:
    def __init__(self, n, t0=1_000_000.0):
        self.frames = [(t0 + i * 0.5, np.full((480, 640, 3), (i * 7) % 255, np.uint8)) for i in range(n)]
        self.status = {"connected": True, "last_frame_ts": time.time(), "reconnects": 0, "last_error": None}

    def latest(self):
        return self.frames.pop(0) if self.frames else None

    def recent_frames(self):
        return []


class Recorder:
    def __init__(self):
        self.sent = []

    def send(self, payload, snap=None, escalation=False):
        self.sent.append((payload, escalation))


@pytest.fixture
def cfg(tmp_path):
    c = copy.deepcopy(CFG)
    c["storage"].update(db=str(tmp_path / "e.db"), media_dir=str(tmp_path / "media"))
    c["cameras"] = [{"id": "cam1", "name": "Lobby", "fps": 2}]
    c["motion"]["enabled"] = False
    return c


def run(cfg, script, n_frames=None, t0=1_000_000.0):
    cam = FakeCamera(n_frames or len(script), t0)
    rec = Recorder()
    store, pipe = build(cfg, detector=FakeDetector(script), cameras={"cam1": cam}, notifier=rec, start=False)
    while pipe.step():
        pass
    return store, pipe, rec


def test_alert_needs_min_hits(cfg):
    store, _, rec = run(cfg, [[PERSON, GUN]] * 2)            # only 2 hits, min_hits = 3
    assert store.list() == [] and rec.sent == []
    store, _, rec = run(cfg, [[PERSON, GUN]] * 3)
    evs = store.list()
    assert len(evs) == 1 and evs[0]["label"] == "handgun" and len(rec.sent) == 1


def test_window_confirmation_for_low_fps(cfg):
    # frames 0.5 s apart in the fake camera; the gun jumps around (no box overlap), as at 0.5 fps with cuts
    jump = [[PERSON, GUN], [Detection("person", 0.9, (400, 100, 500, 400)), Detection("handgun", 0.8, (480, 220, 520, 260))]]
    store, _, _ = run(cfg, jump * 2)                       # box tracking: never 3 overlapping hits -> no alert
    assert store.list() == []
    cfg["alerting"].update(confirm="window", min_hits=2, window_s=6)
    cfg["storage"]["db"] = cfg["storage"]["db"].replace("e.db", "w.db")
    store, _, _ = run(cfg, jump * 2)
    assert len(store.list()) == 1
    store, _, _ = run(cfg, [[PERSON, GUN], [], [], [], [], [], [], [], [], [], [], [], [], [PERSON, GUN]])  # 6.5 s apart
    assert len(store.list()) == 1                          # outside the window: no new alert


def test_person_overlap_rule(cfg):
    store, _, _ = run(cfg, [[PERSON, GUN_FAR]] * 5)
    assert store.list() == []
    cfg["alerting"]["require_person_overlap"] = False
    store, _, _ = run(cfg, [[PERSON, GUN_FAR]] * 5)
    assert len(store.list()) == 1


def test_threshold_and_cooldown(cfg):
    store, _, _ = run(cfg, [[PERSON, PHONE_LIKE]] * 6)
    assert store.list() == []
    # same gun visible for 20 frames (10 s): one alert, not twenty
    store, _, _ = run(cfg, [[PERSON, GUN]] * 20)
    assert len(store.list()) == 1


def test_snapshot_hash_and_chain(cfg):
    cfg["alerting"]["cooldown_s"] = 0
    store, _, _ = run(cfg, [[PERSON, GUN]] * 3 + [[]] * 8 + [[PERSON, GUN]] * 3)
    evs = store.list()
    assert len(evs) == 2
    raw = pathlib.Path(cfg["storage"]["media_dir"]) / f"{evs[0]['id']}_raw.jpg"
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == evs[0]["sha256"]
    assert store.verify_chain() is None
    store.x("UPDATE events SET label='knife' WHERE id=?", (evs[0]["id"],))       # tampering
    assert store.verify_chain()[1] == "record modified"


def test_retention_keeps_chain_verifiable(cfg):
    cfg["alerting"]["cooldown_s"] = 0
    store, _, _ = run(cfg, ([[PERSON, GUN]] * 3 + [[]] * 8) * 3, t0=time.time() - 100 * 86400)  # 100 days ago
    ids = [e["id"] for e in store.list()][::-1]
    assert len(ids) == 3
    store.review(ids[0], "confirmed", "ahmed")
    assert store.purge(30, 365, cfg["storage"]["media_dir"]) == 2                 # confirmed one is kept
    assert store.verify_chain() is None
    cfg2 = copy.deepcopy(cfg)
    cfg2["storage"]["db"] = cfg["storage"]["db"].replace("e.db", "e2.db")
    store2, _, _ = run(cfg2, ([[PERSON, GUN]] * 3 + [[]] * 8) * 3)
    mid = [e["id"] for e in store2.list()][1]
    store2.x("DELETE FROM events WHERE id=?", (mid,))
    assert store2.verify_chain()[1] == "previous record deleted outside retention"


def test_escalation_and_camera_offline(cfg):
    cfg["alerting"]["escalate_after_s"] = 0
    store, pipe, rec = run(cfg, [[PERSON, GUN]] * 3)
    pipe.cameras["cam1"].status["last_frame_ts"] = time.time() - 3600
    pipe.housekeeping()
    kinds = [(p["type"], esc) for p, esc in rec.sent]
    assert ("weapon_detected", True) in kinds and ("camera_offline", False) in kinds


def test_api_auth_and_review(cfg, tmp_path):
    from fastapi.testclient import TestClient
    tok = {"op": "operator-token", "view": "viewer-token"}
    (tmp_path / "tokens.yaml").write_text(yaml.safe_dump({"users": [
        {"name": "sara", "role": "operator", "sha256": hashlib.sha256(tok["op"].encode()).hexdigest()},
        {"name": "ali", "role": "viewer", "sha256": hashlib.sha256(tok["view"].encode()).hexdigest()}]}))
    store, pipe, _ = run(cfg, [[PERSON, GUN]] * 3)
    client = TestClient(create_app(store, pipe, TokenAuth(tmp_path / "tokens.yaml")))
    assert client.get("/api/events").status_code == 401
    evs = client.get("/api/events", headers={"Authorization": "Bearer viewer-token"}).json()
    assert len(evs) == 1
    eid = evs[0]["id"]
    assert client.get(f"/api/events/{eid}/snapshot", headers={"Authorization": "Bearer viewer-token"}).status_code == 200
    r = client.post(f"/api/events/{eid}/review", json={"verdict": "false_alarm"}, headers={"Authorization": "Bearer viewer-token"})
    assert r.status_code == 403
    r = client.post(f"/api/events/{eid}/review", json={"verdict": "false_alarm", "note": "umbrella"},
                    headers={"Authorization": "Bearer operator-token"})
    assert r.status_code == 200 and r.json()["status"] == "false_alarm" and r.json()["reviewed_by"] == "sara"
    assert client.get("/health").json()["cameras_total"] == 1
    assert "cctv_alerts_total" in client.get("/metrics").text
    actions = [a["action"] for a in store.q("SELECT action FROM audit")]
    assert "event_reviewed" in actions and "view_snapshot" in actions


def test_privacy_mask_applied(cfg):
    from vision.video import apply_masks
    img = np.full((100, 100, 3), 255, np.uint8)
    out = apply_masks(img, [[[0, 0], [50, 0], [50, 50], [0, 50]]])
    assert out[10, 10].sum() == 0 and out[90, 90].sum() == 765


def test_clip_writer_works_without_h264(tmp_path, monkeypatch):
    """IBM's ppc64le PyAV has no libx264/mpeg4 encoder: clips must fall back to VP9 WebM and stay playable."""
    import av
    from vision import video
    rng = np.random.default_rng(0)
    frames = [(i * 0.5, rng.integers(0, 255, (120, 160, 3), np.uint8)) for i in range(6)]

    def decoded(path):
        with av.open(path) as c:
            return sum(1 for _ in c.decode(video=0))

    default = video.write_clip(frames, tmp_path / "default.mp4", 2.0)   # whatever this machine has (H.264 on x86)
    assert pathlib.Path(default).exists() and decoded(default) == len(frames)
    monkeypatch.setattr(video, "ENCODERS", [e for e in video.ENCODERS if e[0] == "libvpx-vp9"])
    webm = video.write_clip(frames, tmp_path / "power.mp4", 2.0)      # what a Power server writes
    assert webm.endswith(".webm") and decoded(webm) == len(frames)
