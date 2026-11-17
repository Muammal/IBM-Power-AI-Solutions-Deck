"""traffic_store.py: plate reads, section-speed violations (signed evidence), anonymous traffic statistics."""
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1] / "common" / "python"))
from vision.store import Store  # noqa: E402

SCHEMA = """
CREATE TABLE IF NOT EXISTS reads (
  id TEXT PRIMARY KEY, ts REAL NOT NULL, camera TEXT NOT NULL, lane INTEGER, plate TEXT, plate_key TEXT,
  confidence REAL, agreement REAL, vehicle TEXT, crop TEXT, overview TEXT, crop_sha256 TEXT, overview_sha256 TEXT,
  frame_dt REAL, clock_offset REAL, spot_kmh REAL, in_violation INTEGER DEFAULT 0);
CREATE INDEX IF NOT EXISTS reads_plate ON reads(plate, camera, ts);
CREATE INDEX IF NOT EXISTS reads_ts ON reads(ts);
CREATE TABLE IF NOT EXISTS violations (
  id TEXT PRIMARY KEY, created REAL NOT NULL, section TEXT NOT NULL, plate TEXT NOT NULL,
  entry_read TEXT NOT NULL, exit_read TEXT NOT NULL, measured_kmh REAL, lower_bound_kmh REAL, limit_kmh REAL,
  tolerance_kmh REAL, manifest TEXT NOT NULL, signature TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending',
  reviewer TEXT, reviewed_at REAL, reason TEXT);
CREATE INDEX IF NOT EXISTS violations_status ON violations(status);
CREATE TABLE IF NOT EXISTS stats (
  bin REAL NOT NULL, camera TEXT NOT NULL, lane INTEGER, vehicle TEXT, count INTEGER, speed_sum REAL, speed_n INTEGER,
  PRIMARY KEY (bin, camera, lane, vehicle));
CREATE TABLE IF NOT EXISTS journeys (
  section TEXT, plate_key TEXT, entry_ts REAL, travel_s REAL)
"""
STATS_BIN_S = 900  # 15-minute bins


class TrafficStore(Store):
    def __init__(self, path):
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        super().__init__(path, SCHEMA)

    def add_read(self, r):
        cols = ["id", "ts", "camera", "lane", "plate", "plate_key", "confidence", "agreement", "vehicle", "crop",
                "overview", "crop_sha256", "overview_sha256", "frame_dt", "clock_offset", "spot_kmh"]
        self.x(f"INSERT INTO reads ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})", [r.get(c) for c in cols])
        b = r["ts"] // STATS_BIN_S * STATS_BIN_S
        sp = r.get("spot_kmh")
        self.x("INSERT INTO stats VALUES (?,?,?,?,1,?,?) ON CONFLICT(bin,camera,lane,vehicle) DO UPDATE SET "
               "count=count+1, speed_sum=speed_sum+excluded.speed_sum, speed_n=speed_n+excluded.speed_n",
               (b, r["camera"], r.get("lane"), r.get("vehicle"), sp or 0.0, 1 if sp else 0))

    def find_entry(self, plate, camera, before_ts, max_age_s):
        rows = self.q("SELECT * FROM reads WHERE plate=? AND camera=? AND ts<? AND ts>=? ORDER BY ts DESC LIMIT 1",
                      (plate, camera, before_ts, before_ts - max_age_s))
        return rows[0] if rows else None

    def add_journey(self, section, plate_key, entry_ts, travel_s):
        self.x("INSERT INTO journeys VALUES (?,?,?,?)", (section, plate_key, entry_ts, travel_s))

    def add_violation(self, v):
        self.x("INSERT INTO violations (id, created, section, plate, entry_read, exit_read, measured_kmh, "
               "lower_bound_kmh, limit_kmh, tolerance_kmh, manifest, signature) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
               (v["id"], time.time(), v["section"], v["plate"], v["entry_read"], v["exit_read"], v["measured_kmh"],
                v["lower_bound_kmh"], v["limit_kmh"], v["tolerance_kmh"], json.dumps(v["manifest"], ensure_ascii=False),
                v["signature"]))
        self.x("UPDATE reads SET in_violation=1 WHERE id IN (?,?)", (v["entry_read"], v["exit_read"]))
        self.audit("system", "violation_created", v["id"], {"section": v["section"], "kmh": v["lower_bound_kmh"]})

    def decide(self, vid, decision, reviewer, reason=""):
        if decision not in ("approved", "rejected"):
            raise ValueError("decision must be approved or rejected")
        row = self.q("SELECT status FROM violations WHERE id=?", (vid,))
        if not row:
            raise KeyError(vid)
        if row[0]["status"] != "pending":
            raise PermissionError(f"already {row[0]['status']}")
        self.x("UPDATE violations SET status=?, reviewer=?, reviewed_at=?, reason=? WHERE id=?",
               (decision, reviewer, time.time(), reason[:500], vid))
        self.audit(reviewer, f"violation_{decision}", vid, {"reason": reason[:200]})

    def purge(self, reads_hours, violations_days, media_dir):
        """Plate reads not used by a violation disappear after reads_hours; rejected/old violations later."""
        now, media = time.time(), pathlib.Path(media_dir)
        old_v = self.q("SELECT id, entry_read, exit_read FROM violations WHERE created<? OR (status='rejected' AND created<?)",
                       (now - violations_days * 86400, now - 7 * 86400))
        for v in old_v:
            self.x("DELETE FROM violations WHERE id=?", (v["id"],))
            self.x("UPDATE reads SET in_violation=0 WHERE id IN (?,?)", (v["entry_read"], v["exit_read"]))
        old_r = self.q("SELECT id, crop, overview FROM reads WHERE in_violation=0 AND ts<?", (now - reads_hours * 3600,))
        for r in old_r:
            for f in (r["crop"], r["overview"]):
                if f:
                    (media / pathlib.Path(f).name).unlink(missing_ok=True)
        if old_r:
            self.x("DELETE FROM reads WHERE in_violation=0 AND ts<?", (now - reads_hours * 3600,))
        if old_r or old_v:
            self.audit("system", "retention_purge", None, {"reads": len(old_r), "violations": len(old_v)})
        return len(old_r), len(old_v)
