"""Tests for the situational-awareness service. No network, no LLM: feeds come from fixtures and the LLM is
scripted. Covers: gazetteer, language ID, lookalike domains, KEV matching, collection (dedup, age filter,
conditional GET), LLM-down retry, story clustering, alert rules + dedup, briefing fallback, retention, API roles.

    cd solutions/11-osint-situational-awareness && python -m pytest -q tests
"""
import copy
import hashlib
import json
import pathlib
import sys
import time
from email.utils import formatdate

import httpx
import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from app import CFG, create_app  # noqa: E402
from engine import Engine, OsintStore  # noqa: E402
from enrich import detect_lang, kev_matches, lookalike  # noqa: E402
from gazetteer import find_governorates  # noqa: E402
from vision.auth import TokenAuth  # noqa: E402


def rss(items):
    body = "".join(f"<item><title>{t}</title><link>{u}</link><description>{d}</description>"
                   f"<pubDate>{formatdate(ts)}</pubDate><author>someone@example.com</author></item>"
                   for t, u, d, ts in items)
    return f'<?xml version="1.0" encoding="UTF-8"?><rss version="2.0"><channel><title>x</title>{body}</channel></rss>'


class FakeResponse:
    def __init__(self, text, status=200, headers=None):
        self.text, self.status_code, self.headers = text, status, headers or {}
        self.content = text.encode()

    def json(self):
        return json.loads(self.text)


class FakeHttp:
    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def get(self, url, headers=None, attempts=3):
        self.calls.append((url, headers or {}))
        if headers and headers.get("If-None-Match") == "v1":
            return FakeResponse("", 304)
        for prefix, resp in self.routes.items():
            if url.startswith(prefix):
                return resp
        return None

    def allowed(self, url):
        return True


class FakeLLM:
    def __init__(self, fail=False):
        self.fail, self.calls = fail, 0

    def chat(self, messages, json_schema=None, **kw):
        if self.fail:
            raise httpx.ConnectError("llm down")
        self.calls += 1
        if json_schema is None:
            return "- summary [1]"
        text = messages[-1]["content"]
        if "football" in text.lower():
            return {"is_event": False, "category": "other", "governorates": [], "severity": 1, "summary_ar": "",
                    "summary_en": "", "confidence": 0.9}
        cat = "infrastructure_outage" if ("electricity" in text or "كهرباء" in text) else "security_incident"
        return {"is_event": True, "category": cat, "governorates": [], "severity": 4, "organizations": ["Ministry of Electricity"],
                "location_text": "", "summary_ar": "ملخص", "summary_en": "summary", "confidence": 0.8}


NOW = time.time()


@pytest.fixture
def cfg(tmp_path):
    c = copy.deepcopy(CFG)
    c["storage"]["db"] = str(tmp_path / "o.db")
    c["sources"] = [{"id": "news", "type": "rss", "url": "https://news.example/rss", "lang": "ar", "interval_min": 10}]
    c["organisation"]["software_inventory"] = str(tmp_path / "inv.csv")
    (tmp_path / "inv.csv").write_text("vendor,product\nfortinet,fortios\n")
    c["alerts"] = [{"name": "outage", "when": {"categories": ["infrastructure_outage"], "min_severity": 3}, "notify": ["ops", "soc"]},
                   {"name": "basra-security", "when": {"categories": ["security_incident"], "governorates": ["Basra"]}, "notify": ["soc"]}]
    c["channels"] = {"ops": "https://hooks.example/ops", "soc": "https://hooks.example/soc"}
    return c


FEED = rss([
    ("انقطاع الكهرباء في البصرة بسبب ارتفاع الحرارة", "https://news.example/1", "توقف محطات في البصرة والزبير", NOW - 3600),
    ("انقطاع واسع للكهرباء في البصرة مع ارتفاع درجات الحرارة", "https://news.example/2", "البصرة", NOW - 1800),
    ("Football: Iraq beats Jordan", "https://news.example/3", "sports", NOW - 600),
    ("Old news from last month", "https://news.example/4", "", NOW - 40 * 86400),
])


def make(cfg, llm=None, feed=FEED):
    sent = []
    http = FakeHttp({"https://news.example/rss": FakeResponse(feed, headers={"etag": "v1"})})
    eng = Engine(cfg, OsintStore(cfg["storage"]["db"]), llm or FakeLLM(), http=http,
                 post=lambda url, payload: sent.append((url, payload)) or True)
    return eng, http, sent


def test_gazetteer_and_language():
    assert find_governorates("انفجار في الموصل وتلعفر") == ["Nineveh"]
    assert set(find_governorates("Flooding in Erbil and Sulaymaniyah")) == {"Erbil", "Sulaymaniyah"}
    assert find_governorates("وبغداد تشهد") == ["Baghdad"]
    assert detect_lang("هەولێر شارێکی جوانە لە کوردستان") == "ckb"
    assert detect_lang("بغداد عاصمة العراق") == "ar" and detect_lang("Baghdad is the capital") == "en"


def test_lookalike_and_kev():
    org = {"domains": ["example-bank.iq"], "brand_keywords": ["examplebank"]}
    assert lookalike("examp1e-bank.com", org)
    assert lookalike("login-examplebank.net", org)
    assert lookalike("www.example-bank.iq", org) is None and lookalike("weather.com", org) is None
    inv = [{"vendor": "fortinet", "product": "fortios"}]
    assert kev_matches({"vendor": "Fortinet", "product": "FortiOS"}, inv)
    assert not kev_matches({"vendor": "Microsoft", "product": "Exchange"}, inv)


def test_collect_dedup_age_and_conditional_get(cfg):
    eng, http, _ = make(cfg)
    assert eng.collect(force=True) == 3                       # 4 items, 1 too old
    assert eng.collect(force=True) == 0                       # 304 Not Modified via ETag
    assert http.calls[-1][1].get("If-None-Match") == "v1"
    assert not any("someone@example.com" in json.dumps(r) for r in eng.store.q("SELECT * FROM items"))  # authors dropped


def test_llm_down_keeps_items_pending_then_recovers(cfg):
    eng, _, _ = make(cfg, llm=FakeLLM(fail=True))
    eng.collect(force=True)
    assert eng.process() == 0 and len(eng.pending()) == 3
    eng.llm = FakeLLM()
    assert eng.process() == 2                                  # football item is not an event
    assert len(eng.pending()) == 0
    assert eng.store.q("SELECT status FROM items WHERE url='https://news.example/3'")[0]["status"] == "skipped"


def test_clustering_rules_and_dedup(cfg):
    eng, _, sent = make(cfg)
    eng.collect(force=True)
    eng.process()
    evs = eng.store.q("SELECT * FROM events")
    assert len(evs) == 2 and len({e["story_id"] for e in evs}) == 1           # same story, two outlets
    assert json.loads(evs[0]["governorates"]) == ["Basra"]                      # gazetteer filled what the LLM left empty
    assert sorted(u for u, _ in sent) == ["https://hooks.example/ops", "https://hooks.example/soc"]  # one per channel
    assert eng.store.q("SELECT count(*) n FROM alerts")[0]["n"] == 2


def test_briefing_without_llm(cfg):
    eng, _, _ = make(cfg)
    eng.collect(force=True)
    eng.process()
    eng.llm = FakeLLM(fail=True)
    md = eng.briefing("2026-11-16")
    assert "Infrastructure Outage" in md and "https://news.example/1" in md and "Executive summary" not in md


def test_retention(cfg):
    eng, _, _ = make(cfg)
    eng.collect(force=True)
    eng.process()
    assert eng.purge(now=NOW + 400 * 86400) == 2
    assert eng.store.q("SELECT count(*) n FROM items")[0]["n"] == 0


def test_api(cfg, tmp_path):
    from fastapi.testclient import TestClient
    (tmp_path / "tokens.yaml").write_text(yaml.safe_dump({"users": [
        {"name": "noor", "role": "operator", "sha256": hashlib.sha256(b"op").hexdigest()},
        {"name": "guest", "role": "viewer", "sha256": hashlib.sha256(b"v").hexdigest()}]}))
    eng, _, _ = make(cfg)
    eng.collect(force=True)
    eng.process()
    c = TestClient(create_app(eng, TokenAuth(tmp_path / "tokens.yaml")))
    V, O = {"Authorization": "Bearer v"}, {"Authorization": "Bearer op"}
    assert c.get("/api/events").status_code == 401
    evs = c.get("/api/events?governorate=Basra&min_severity=3", headers=V).json()
    assert len(evs) == 2
    assert len(c.get("/api/search", params={"q": "الكهرباء"}, headers=V).json()) >= 1
    eid = evs[0]["id"]
    assert c.post(f"/api/events/{eid}/correction", json={"severity": 2}, headers=V).status_code == 403
    r = c.post(f"/api/events/{eid}/correction", json={"severity": 2, "governorates": ["Basra"]}, headers=O)
    assert r.status_code == 200 and r.json()["corrected"]["by"] == "noor"
    assert c.post(f"/api/events/{eid}/correction", json={"governorates": ["Atlantis"]}, headers=O).status_code == 400
    assert c.get("/health").json()["sources"] == 1
    paths = {r.path for r in c.app.routes}
    assert not any(k in p for p in paths for k in ("person", "people", "profile", "user_search"))
