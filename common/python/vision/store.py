"""store.py: SQLite helper for single-node services (WAL mode, one connection per thread, append-only audit log).

SQLite in WAL mode comfortably handles the write rate of an event system (alerts, plate reads at a few per second).
For multi-node deployments put the same schema on PostgreSQL (the SQL used here is standard).
"""
import json
import sqlite3
import threading
import time

AUDIT_SQL = """CREATE TABLE IF NOT EXISTS audit (
  id INTEGER PRIMARY KEY, ts REAL NOT NULL, actor TEXT NOT NULL, action TEXT NOT NULL, target TEXT, detail TEXT)"""


class Store:
    def __init__(self, path, schema_sql):
        self.path = str(path)
        self._local = threading.local()
        with self.conn() as c:
            c.executescript(schema_sql + ";\n" + AUDIT_SQL)

    def conn(self):
        c = getattr(self._local, "c", None)
        if c is None:
            c = sqlite3.connect(self.path, timeout=30, isolation_level=None, check_same_thread=False)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA synchronous=NORMAL")
            c.execute("PRAGMA foreign_keys=ON")
            self._local.c = c
        return c

    def q(self, sql, args=()):
        return [dict(r) for r in self.conn().execute(sql, args).fetchall()]

    def x(self, sql, args=()):
        cur = self.conn().execute(sql, args)
        return cur.lastrowid

    def audit(self, actor, action, target=None, detail=None):
        self.x("INSERT INTO audit (ts, actor, action, target, detail) VALUES (?,?,?,?,?)",
               (time.time(), actor, action, str(target) if target is not None else None,
                json.dumps(detail, ensure_ascii=False) if detail is not None else None))
