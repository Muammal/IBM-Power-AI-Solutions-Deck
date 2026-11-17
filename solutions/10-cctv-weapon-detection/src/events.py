"""events.py: alert/event storage with a tamper-evident hash chain, reviews, and retention."""
import json
import pathlib
import shutil
import sys
import threading
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1] / "common" / "python"))
from vision.evidence import chain_hash  # noqa: E402
from vision.store import Store  # noqa: E402

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
  id TEXT PRIMARY KEY, ts REAL NOT NULL, camera TEXT NOT NULL, label TEXT NOT NULL, score REAL, box TEXT,
  snapshot TEXT, clip TEXT, sha256 TEXT, prev TEXT, chain TEXT, status TEXT NOT NULL DEFAULT 'new',
  reviewed_by TEXT, reviewed_at REAL, note TEXT, notified INTEGER DEFAULT 0, escalated INTEGER DEFAULT 0);
CREATE INDEX IF NOT EXISTS events_ts ON events(ts);
CREATE INDEX IF NOT EXISTS events_status ON events(status);
CREATE TABLE IF NOT EXISTS system_alerts (
  id INTEGER PRIMARY KEY, ts REAL NOT NULL, camera TEXT, kind TEXT NOT NULL, detail TEXT)
"""
VERDICTS = {"confirmed", "false_alarm"}


class EventStore(Store):
    def __init__(self, path):
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        super().__init__(path, SCHEMA)
        self._chain_lock = threading.Lock()

    @staticmethod
    def _record(ev):
        """The fields protected by the hash chain (review status is mutable and audited separately)."""
        box = json.loads(ev["box"]) if isinstance(ev["box"], str) else ev["box"]
        return {"id": ev["id"], "ts": ev["ts"], "camera": ev["camera"], "label": ev["label"],
                "score": ev["score"], "box": box, "sha256": ev["sha256"]}

    def _purged_chains(self):
        out = []
        for r in self.q("SELECT detail FROM audit WHERE action='retention_purge' ORDER BY id"):
            out += json.loads(r["detail"]).get("chains", [])
        return out

    def _purged_tail(self):
        p = self._purged_chains()
        return p[-1] if p else ""

    def add(self, ev):
        with self._chain_lock:  # one writer at a time extends the chain
            last = self.q("SELECT chain FROM events ORDER BY rowid DESC LIMIT 1")
            ev["prev"] = last[0]["chain"] if last else self._purged_tail()
            ev["chain"] = chain_hash(ev["prev"], self._record(ev))
            self.x("INSERT INTO events (id, ts, camera, label, score, box, snapshot, clip, sha256, prev, chain) "
                   "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                   (ev["id"], ev["ts"], ev["camera"], ev["label"], ev["score"], json.dumps(ev["box"]),
                    ev.get("snapshot"), ev.get("clip"), ev["sha256"], ev["prev"], ev["chain"]))
        self.audit("system", "event_created", ev["id"], {"camera": ev["camera"], "label": ev["label"]})
        return ev

    def get(self, event_id):
        r = self.q("SELECT * FROM events WHERE id=?", (event_id,))
        return r[0] if r else None

    def list(self, status=None, limit=100, since=None):
        sql, args = "SELECT * FROM events WHERE 1=1", []
        if status:
            sql += " AND status=?"
            args.append(status)
        if since:
            sql += " AND ts>=?"
            args.append(since)
        return self.q(sql + " ORDER BY ts DESC LIMIT ?", (*args, int(limit)))

    def review(self, event_id, verdict, user, note=""):
        if verdict not in VERDICTS:
            raise ValueError(f"verdict must be one of {VERDICTS}")
        if not self.get(event_id):
            raise KeyError(event_id)
        self.x("UPDATE events SET status=?, reviewed_by=?, reviewed_at=?, note=? WHERE id=?",
               (verdict, user, time.time(), note[:1000], event_id))
        self.audit(user, "event_reviewed", event_id, {"verdict": verdict, "note": note[:200]})

    def set_clip(self, event_id, clip):
        self.x("UPDATE events SET clip=? WHERE id=?", (clip, event_id))

    def system_alert(self, camera, kind, detail):
        self.x("INSERT INTO system_alerts (ts, camera, kind, detail) VALUES (?,?,?,?)", (time.time(), camera, kind, detail))

    def verify_chain(self):
        """Returns None if intact, else (event_id, reason).
        Each record must hash correctly (detects edits), and its `prev` must be the previous remaining record or a
        chain value removed by a logged retention purge (detects silent deletions)."""
        purged = set(self._purged_chains()) | {""}
        prev = None
        for r in self.q("SELECT * FROM events ORDER BY rowid ASC"):
            if chain_hash(r["prev"], self._record(r)) != r["chain"]:
                return r["id"], "record modified"
            if r["prev"] != prev and r["prev"] not in purged:
                return r["id"], "previous record deleted outside retention"
            prev = r["chain"]
        return None

    def purge(self, retention_days, confirmed_days, media_dir, negatives_dir=None):
        """Delete expired events and their media. False alarms can be kept as training negatives first."""
        now = time.time()
        old = self.q("SELECT * FROM events WHERE (status!='confirmed' AND ts<?) OR (status='confirmed' AND ts<?)",
                     (now - retention_days * 86400, now - confirmed_days * 86400))
        for r in old:
            if negatives_dir and r["status"] == "false_alarm" and r["snapshot"]:
                pathlib.Path(negatives_dir).mkdir(parents=True, exist_ok=True)
                raw = pathlib.Path(media_dir) / f"{r['id']}_raw.jpg"
                if raw.exists():
                    shutil.copy2(raw, pathlib.Path(negatives_dir) / raw.name)
            for f in pathlib.Path(media_dir).glob(f"{r['id']}*"):
                f.unlink(missing_ok=True)
        # Log the removed chain values so verify_chain() can tell a lawful purge from tampering.
        if old:
            self.x("DELETE FROM events WHERE id IN (%s)" % ",".join("?" * len(old)), [r["id"] for r in old])
            self.audit("system", "retention_purge", None, {"count": len(old), "chains": [r["chain"] for r in old]})
        return len(old)
