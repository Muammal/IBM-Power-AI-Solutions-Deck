"""engine.py: storage + the collect -> enrich -> cluster -> alert -> brief loop."""
import datetime as dt
import json
import logging
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1] / "common" / "python"))
sys.path.insert(0, str(ROOT / "src"))
import httpx  # noqa: E402
from prometheus_client import Counter, Gauge  # noqa: E402
from vision.store import Store  # noqa: E402  (generic SQLite helper, shared with the CV solutions)

import collectors as C  # noqa: E402
from enrich import detect_lang, extract_event, jaccard, kev_matches, load_inventory, lookalike, shingles  # noqa: E402

log = logging.getLogger("osint.engine")
ITEMS = Counter("osint_items_collected_total", "New items collected", ["source"])
EVENTS = Counter("osint_events_total", "Events extracted", ["category"])
ALERTS = Counter("osint_alerts_total", "Alerts sent", ["rule"])
PENDING = Gauge("osint_items_pending", "Items waiting for enrichment")

SCHEMA = """
CREATE TABLE IF NOT EXISTS items (id TEXT PRIMARY KEY, source TEXT, url TEXT, title TEXT, text TEXT, published REAL,
  lang TEXT, collected REAL, status TEXT DEFAULT 'pending', error TEXT, extra TEXT);
CREATE INDEX IF NOT EXISTS items_status ON items(status);
CREATE TABLE IF NOT EXISTS events (id TEXT PRIMARY KEY, item_id TEXT, published REAL, category TEXT,
  governorates TEXT, severity INTEGER, organizations TEXT, location_text TEXT, summary_ar TEXT, summary_en TEXT,
  confidence REAL, story_id INTEGER, url TEXT, title TEXT, source TEXT, corrected TEXT);
CREATE INDEX IF NOT EXISTS events_pub ON events(published);
CREATE TABLE IF NOT EXISTS stories (id INTEGER PRIMARY KEY, category TEXT, title TEXT, shingles TEXT,
  first_seen REAL, last_seen REAL, n_items INTEGER DEFAULT 1, max_severity INTEGER);
CREATE TABLE IF NOT EXISTS alerts (id INTEGER PRIMARY KEY, ts REAL, rule TEXT, event_id TEXT, story_id INTEGER,
  channel TEXT, delivered INTEGER, UNIQUE(rule, story_id, channel));
CREATE TABLE IF NOT EXISTS sources (id TEXT PRIMARY KEY, state TEXT, last_run REAL, last_ok REAL, items_total INTEGER DEFAULT 0,
  last_error TEXT);
CREATE TABLE IF NOT EXISTS briefings (day TEXT PRIMARY KEY, created REAL, markdown TEXT)
"""


class OsintStore(Store):
    def __init__(self, path):
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        super().__init__(path, SCHEMA)
        try:
            self.x("CREATE VIRTUAL TABLE IF NOT EXISTS events_fts USING fts5(event_id UNINDEXED, title, summary_ar, summary_en)")
            self.fts = True
        except Exception:  # SQLite built without FTS5: fall back to LIKE search
            self.fts = False

    def search(self, q, limit=50):
        if self.fts:
            ids = [r["event_id"] for r in self.q("SELECT event_id FROM events_fts WHERE events_fts MATCH ? LIMIT ?",
                                                 (q, limit))]
            return [self.q("SELECT * FROM events WHERE id=?", (i,))[0] for i in ids]
        like = f"%{q}%"
        return self.q("SELECT * FROM events WHERE title LIKE ? OR summary_ar LIKE ? OR summary_en LIKE ? LIMIT ?",
                      (like, like, like, limit))


class Engine:
    def __init__(self, cfg, store, llm, http=None, post=None):
        self.cfg, self.store, self.llm = cfg, store, llm
        self.http = http or C.Http(cfg)
        self._post = post or self._webhook
        self.inventory = load_inventory(cfg["organisation"]["software_inventory"])

    # ---------------------------------------------------------------- collection
    def collect(self, now=None, force=False):
        now = now or time.time()
        new = 0
        for src in self.cfg["sources"]:
            row = self.store.q("SELECT * FROM sources WHERE id=?", (src["id"],))
            row = row[0] if row else {"state": "{}", "last_run": 0}
            if not force and now - (row["last_run"] or 0) < float(src.get("interval_min", 15)) * 60:
                continue
            state = json.loads(row["state"] or "{}")
            try:
                kind = src["type"]
                if kind == "rss":
                    items, state = C.collect_rss(src, self.http, state)
                elif kind == "gdelt":
                    items, state = C.collect_gdelt(src, self.http, state)
                elif kind == "kev":
                    items, state = C.collect_kev(src, self.http, state)
                elif kind == "crtsh":
                    items, state = C.collect_crtsh(src, self.http, state, self.cfg["organisation"])
                else:
                    raise ValueError(f"unknown source type {kind}")
                n = self._ingest(src, items, now)
                new += n
                stale = None
                if items and src["type"] in ("rss", "gdelt"):
                    newest = max(i["published"] for i in items)
                    if now - newest > float(self.cfg["collection"]["max_item_age_days"]) * 86400:
                        stale = f"feed stale: newest item {time.strftime('%Y-%m-%d', time.gmtime(newest))}"
                        log.warning("source %s: %s", src["id"], stale)
                self.store.x("INSERT INTO sources (id, state, last_run, last_ok, items_total) VALUES (?,?,?,?,?) "
                             "ON CONFLICT(id) DO UPDATE SET state=excluded.state, last_run=excluded.last_run, "
                             "last_ok=excluded.last_ok, items_total=items_total+?, last_error=excluded.last_error",
                             (src["id"], json.dumps(state), now, now, n, n))
                self.store.x("UPDATE sources SET last_error=? WHERE id=?", (stale, src["id"]))
            except Exception as exc:
                log.error("source %s failed: %s", src["id"], exc)
                self.store.x("INSERT INTO sources (id, last_run, last_error) VALUES (?,?,?) ON CONFLICT(id) DO UPDATE "
                             "SET last_run=excluded.last_run, last_error=excluded.last_error", (src["id"], now, str(exc)[:300]))
        return new

    def _ingest(self, src, items, now):
        max_age = float(self.cfg["collection"]["max_item_age_days"]) * 86400
        n = 0
        for it in items:
            iid = C.item_id(it["url"])
            too_old = now - it["published"] > max_age and "kev" not in it   # KEV: old entries still matter
            if too_old or self.store.q("SELECT 1 FROM items WHERE id=?", (iid,)):
                continue
            if self.cfg["collection"].get("fetch_full_text") and src["type"] in ("rss", "gdelt"):
                it = C.fetch_full_text(it, self.http)
            it["lang"] = it.get("lang") or detect_lang(it["title"] + " " + it["text"])
            extra = {k: it[k] for k in ("kev", "domain") if k in it}
            self.store.x("INSERT INTO items (id, source, url, title, text, published, lang, collected, extra) VALUES (?,?,?,?,?,?,?,?,?)",
                         (iid, it["source"], it["url"], it["title"], it["text"], it["published"], it["lang"], now,
                          json.dumps(extra)))
            ITEMS.labels(it["source"]).inc()
            n += 1
        return n

    # ---------------------------------------------------------------- enrichment
    def process(self, limit=50):
        done = 0
        for it in self.store.q("SELECT * FROM items WHERE status='pending' ORDER BY published DESC LIMIT ?", (limit,)):
            extra = json.loads(it["extra"] or "{}")
            try:
                if "kev" in extra:
                    ev = self._kev_event(it, extra["kev"])
                elif "domain" in extra:
                    ev = self._ct_event(it, extra["domain"])
                else:
                    ev = extract_event(it, self.llm)
            except (httpx.HTTPError, OSError) as exc:          # LLM down: keep pending, retry later
                log.warning("enrichment unavailable (%s); %d items stay pending", exc, len(self.pending()))
                break
            except Exception as exc:                           # bad output for this item: record and move on
                self.store.x("UPDATE items SET status='error', error=? WHERE id=?", (str(exc)[:300], it["id"]))
                continue
            if not ev or not ev.get("is_event"):
                self.store.x("UPDATE items SET status='skipped' WHERE id=?", (it["id"],))
                continue
            self._save_event(it, ev)
            self.store.x("UPDATE items SET status='done' WHERE id=?", (it["id"],))
            done += 1
        PENDING.set(len(self.pending()))
        return done

    def pending(self):
        return self.store.q("SELECT id FROM items WHERE status='pending'")

    def _kev_event(self, it, kev):
        hits = kev_matches(kev, self.inventory)
        if not hits:
            return None                       # catalogue entry irrelevant to our estate: no event
        return {"is_event": True, "category": "cyber", "governorates": ["Nationwide"], "severity": 5 if kev.get("ransomware") == "Known" else 4, "organizations": [kev["vendor"]], "location_text": "",
            "summary_en": f"Actively exploited vulnerability {kev['cve']} affects {kev['vendor']} {kev['product']} used in our estate.",
            "summary_ar": f"ثغرة مستغلة فعلياً {kev['cve']} تؤثر على {kev['vendor']} {kev['product']} المستخدم لدينا.",
            "confidence": 1.0}

    def _ct_event(self, it, domain):
        reason = lookalike(domain, self.cfg["organisation"])
        if not reason:
            return None
        return {"is_event": True, "category": "cyber", "governorates": ["Nationwide"], "severity": 4,
                "organizations": [], "location_text": domain,
                "summary_en": f"Possible phishing domain {domain} ({reason}) received a TLS certificate.",
                "summary_ar": f"نطاق احتيالي محتمل {domain} ({reason}) حصل على شهادة TLS.", "confidence": 0.9}

    def _save_event(self, it, ev):
        story = self._cluster(it, ev)
        self.store.x("INSERT OR REPLACE INTO events (id, item_id, published, category, governorates, severity, organizations, "
                     "location_text, summary_ar, summary_en, confidence, story_id, url, title, source) VALUES "
                     "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                     (it["id"], it["id"], it["published"], ev["category"], json.dumps(ev["governorates"]), int(ev["severity"]),
                      json.dumps(ev.get("organizations", []), ensure_ascii=False), ev.get("location_text", ""),
                      ev["summary_ar"], ev["summary_en"], float(ev["confidence"]), story, it["url"], it["title"], it["source"]))
        if self.store.fts:
            self.store.x("INSERT INTO events_fts VALUES (?,?,?,?)", (it["id"], it["title"], ev["summary_ar"], ev["summary_en"]))
        EVENTS.labels(ev["category"]).inc()
        self._rules(it, ev, story)

    def _cluster(self, it, ev):
        sh = shingles(it["title"])
        since = it["published"] - float(self.cfg["clustering"]["window_hours"]) * 3600
        best, best_j = None, float(self.cfg["clustering"]["jaccard"])
        for s in self.store.q("SELECT * FROM stories WHERE category=? AND last_seen>=?", (ev["category"], since)):
            j = jaccard(sh, set(json.loads(s["shingles"])))
            if j >= best_j:
                best, best_j = s, j
        if best:
            self.store.x("UPDATE stories SET last_seen=max(last_seen,?), n_items=n_items+1, max_severity=max(max_severity,?) "
                         "WHERE id=?", (it["published"], int(ev["severity"]), best["id"]))
            return best["id"]
        return self.store.x("INSERT INTO stories (category, title, shingles, first_seen, last_seen, max_severity) VALUES (?,?,?,?,?,?)",
                            (ev["category"], it["title"], json.dumps(sorted(sh), ensure_ascii=False), it["published"],
                             it["published"], int(ev["severity"])))

    # ---------------------------------------------------------------- alerting
    def _rules(self, it, ev, story):
        org = self.cfg["organisation"]
        for rule in self.cfg.get("alerts", []):
            w = rule["when"]
            if w.get("categories") and ev["category"] not in w["categories"]:
                continue
            if w.get("governorates") and not set(w["governorates"]) & set(ev["governorates"]):
                continue
            if int(ev["severity"]) < int(w.get("min_severity", 1)):
                continue
            if w.get("keywords_from_org"):
                text = (it["title"] + " " + it["text"]).lower()
                if not any(k.lower() in text for k in org.get("brand_keywords", []) + [org["name"].lower()]):
                    continue
            for ch in rule.get("notify", []):
                try:
                    self.store.x("INSERT INTO alerts (ts, rule, event_id, story_id, channel, delivered) VALUES (?,?,?,?,?,0)",
                                 (time.time(), rule["name"], it["id"], story, ch))
                except Exception:      # UNIQUE(rule, story, channel): this story was already sent there
                    continue
                payload = {"rule": rule["name"], "category": ev["category"], "severity": ev["severity"],
                           "governorates": ev["governorates"], "summary_en": ev["summary_en"], "summary_ar": ev["summary_ar"],
                           "url": it["url"], "source": it["source"],
                           "published": dt.datetime.fromtimestamp(it["published"], dt.timezone.utc).isoformat()}
                ok = self._post(self.cfg["channels"].get(ch), payload)
                self.store.x("UPDATE alerts SET delivered=? WHERE rule=? AND story_id=? AND channel=?",
                             (1 if ok else 0, rule["name"], story, ch))
                ALERTS.labels(rule["name"]).inc()

    def _webhook(self, url, payload):
        if not url:
            return False
        for i in range(3):
            try:
                httpx.post(url, json=payload, timeout=10).raise_for_status()
                return True
            except httpx.HTTPError:
                time.sleep(2 ** i)
        return False

    # ---------------------------------------------------------------- briefing
    def briefing(self, day=None, now=None):
        now = now or time.time()
        day = day or dt.date.fromtimestamp(now).isoformat()
        since = now - float(self.cfg["briefing"]["lookback_hours"]) * 3600
        evs = self.store.q("SELECT * FROM events WHERE published>=? ORDER BY severity DESC, published DESC LIMIT 60", (since,))
        lines = [f"# Situational briefing / الموجز اليومي — {day}", "",
                 f"{len(evs)} events in the last {self.cfg['briefing']['lookback_hours']} h.", ""]
        by_cat = {}
        for e in evs:
            by_cat.setdefault(e["category"], []).append(e)
        for cat, items in sorted(by_cat.items(), key=lambda kv: -max(x["severity"] for x in kv[1])):
            lines.append(f"## {cat.replace('_', ' ').title()}")
            for e in items[:8]:
                govs = ", ".join(json.loads(e["governorates"])) or "-"
                lines.append(f"- **S{e['severity']}** [{govs}] {e['summary_en']}  \n  {e['summary_ar']}  \n  source: {e['url']}")
            lines.append("")
        md = "\n".join(lines)
        if evs:
            try:   # optional executive summary on top; the factual list above never depends on the LLM
                digest = "\n".join(f"[{i + 1}] S{e['severity']} {e['category']} {e['governorates']}: {e['summary_en']}"
                                   for i, e in enumerate(evs[:30]))
                summary = self.llm.chat([{"role": "user", "content":
                    "Write a 5-bullet executive summary in English, then the same in Arabic, of these events for the "
                    "security and operations leadership of an organisation in Iraq. Cite items as [n]. Use only these "
                    f"facts.\n\n{digest}"}], max_tokens=900, temperature=0.2)
                md = md.replace("\n\n", f"\n\n## Executive summary\n\n{summary}\n\n", 1)
            except Exception as exc:
                log.warning("briefing summary unavailable: %s", exc)
        self.store.x("INSERT OR REPLACE INTO briefings VALUES (?,?,?)", (day, now, md))
        return md

    # ---------------------------------------------------------------- retention
    def purge(self, now=None):
        now = now or time.time()
        r = self.cfg["retention"]
        old_items = now - float(r["items_days"]) * 86400
        old_events = now - float(r["events_days"]) * 86400
        self.store.x("DELETE FROM items WHERE collected<?", (old_items,))
        ids = [e["id"] for e in self.store.q("SELECT id FROM events WHERE published<?", (old_events,))]
        for i in ids:
            self.store.x("DELETE FROM events WHERE id=?", (i,))
            if self.store.fts:
                self.store.x("DELETE FROM events_fts WHERE event_id=?", (i,))
        return len(ids)
