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
