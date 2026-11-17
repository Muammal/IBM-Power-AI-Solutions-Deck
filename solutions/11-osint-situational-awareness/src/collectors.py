"""collectors.py: polite, resilient collection from public sources.

Every collector returns Item dicts: {source, url, title, text, published (epoch), lang}. Politeness rules:
  * identifying User-Agent with a contact address
  * conditional GET (ETag / Last-Modified) so unchanged feeds cost nothing
  * minimum delay per host (GDELT requires >= 5 s), retries with backoff, timeouts
  * full-text download is off by default and, when on, honours robots.txt
Only titles, summaries and links are kept: authors and any personal fields of feeds are dropped.
"""
import calendar
import hashlib
import html
import json
import logging
import re
import threading
import time
import urllib.parse
import urllib.robotparser

import feedparser
import httpx

log = logging.getLogger("osint.collect")
_TAGS = re.compile(r"<[^>]+>")


def clean(text):
    return re.sub(r"\s+", " ", html.unescape(_TAGS.sub(" ", text or ""))).strip()


def item_id(url):
    return hashlib.sha256(url.encode()).hexdigest()[:24]


class Http:
    def __init__(self, cfg):
        c = cfg["collection"]
        self.client = httpx.Client(timeout=c["timeout_s"], follow_redirects=True, headers={"User-Agent": c["user_agent"]})
        self.min_gap = float(c["min_seconds_between_requests_per_host"])
        self.last = {}
        self.lock = threading.Lock()
        self.robots = {}

    def get(self, url, headers=None, attempts=3):
        host = urllib.parse.urlparse(url).netloc
        for i in range(attempts):
            with self.lock:
                wait = self.last.get(host, 0) + self.min_gap - time.time()
                if wait > 0:
                    time.sleep(wait)
                self.last[host] = time.time()
            try:
                r = self.client.get(url, headers=headers or {})
                if r.status_code == 429 or r.status_code >= 500:
                    raise httpx.HTTPStatusError(f"HTTP {r.status_code}", request=r.request, response=r)
                return r
            except httpx.HTTPError as exc:
                log.warning("GET %s failed (%d/%d): %s", url, i + 1, attempts, exc)
                time.sleep(min(60, 5 * 2 ** i))
        return None

    def allowed(self, url):
        p = urllib.parse.urlparse(url)
        base = f"{p.scheme}://{p.netloc}"
        if base not in self.robots:
            rp = urllib.robotparser.RobotFileParser()
            r = self.get(base + "/robots.txt", attempts=1)
            rp.parse(r.text.splitlines() if r is not None and r.status_code == 200 else [])
            self.robots[base] = rp
        return self.robots[base].can_fetch(self.client.headers["User-Agent"], url)


class SourceUnavailable(RuntimeError):
    pass


def _need(r, url):
    if r is None:
        raise SourceUnavailable(f"unreachable after retries: {url}")
    return r


def collect_rss(src, http, state):
    headers = {}
    if state.get("etag"):
        headers["If-None-Match"] = state["etag"]
    if state.get("modified"):
        headers["If-Modified-Since"] = state["modified"]
    r = _need(http.get(src["url"], headers), src["url"])
    if r.status_code == 304:
        return [], state
    state = {"etag": r.headers.get("etag"), "modified": r.headers.get("last-modified")}
    feed = feedparser.parse(r.content)
    items = []
    for e in feed.entries:
        url = e.get("link")
        if not url:
            continue
        ts = e.get("published_parsed") or e.get("updated_parsed")
        items.append({"source": src["id"], "url": url, "title": clean(e.get("title")),
                      "text": clean(e.get("summary") or e.get("description"))[:4000],
                      "published": calendar.timegm(ts) if ts else time.time(), "lang": src.get("lang")})
    return items, state


def collect_gdelt(src, http, state):
    q = urllib.parse.quote(src["query"])
    url = f"https://api.gdeltproject.org/api/v2/doc/doc?query={q}&mode=artlist&format=json&maxrecords=75&timespan=2h&sort=datedesc"
    r = _need(http.get(url), "GDELT")
    if r.status_code != 200:
        raise SourceUnavailable(f"GDELT HTTP {r.status_code}")
    try:
        arts = r.json().get("articles", [])
    except json.JSONDecodeError:  # GDELT sends rate-limit notices as plain text, sometimes with HTTP 200
        raise SourceUnavailable(f"GDELT non-JSON answer: {r.text[:120]}")
    items = []
    for a in arts:
        try:
            ts = calendar.timegm(time.strptime(a["seendate"], "%Y%m%dT%H%M%SZ"))
        except (KeyError, ValueError):
            ts = time.time()
        lang = {"Arabic": "ar", "English": "en", "Kurdish": "ckb"}.get(a.get("language"), None)
        items.append({"source": src["id"], "url": a["url"], "title": clean(a.get("title")), "text": "",
                      "published": ts, "lang": lang})
    return items, state


def collect_kev(src, http, state):
    r = _need(http.get(src["url"]), src["url"])
    if r.status_code != 200:
        raise SourceUnavailable(f"KEV HTTP {r.status_code}")
    data = r.json()
    since = state.get("last_date_added", "")
    items = []
    for v in data.get("vulnerabilities", []):
        if v["dateAdded"] <= since:
            continue
        url = f"https://nvd.nist.gov/vuln/detail/{v['cveID']}"
        items.append({"source": src["id"], "url": url, "lang": "en",
                      "title": f"{v['cveID']}: {v['vendorProject']} {v['product']} - {v['vulnerabilityName']}",
                      "text": f"{v['shortDescription']} Required action: {v['requiredAction']} Due: {v['dueDate']}",
                      "published": calendar.timegm(time.strptime(v["dateAdded"], "%Y-%m-%d")),
                      "kev": {"cve": v["cveID"], "vendor": v["vendorProject"], "product": v["product"],
                              "ransomware": v.get("knownRansomwareCampaignUse")}})
    dates = [v["dateAdded"] for v in data.get("vulnerabilities", [])]
    return items, {"last_date_added": max(dates + [since])}


def collect_crtsh(src, http, state, org):
    """Certificates recently issued for names containing the brand keywords (public Certificate Transparency)."""
    items = []
    for kw in org.get("brand_keywords", []):
        r = http.get(f"https://crt.sh/?q=%25{urllib.parse.quote(kw)}%25&output=json&exclude=expired", attempts=2)
        if r is None or r.status_code != 200:
            continue
        try:
            rows = r.json()
        except json.JSONDecodeError:
            continue
        for row in rows:
            for name in set(row.get("name_value", "").lower().split("\n")):
                url = f"https://crt.sh/?id={row.get('id')}#{name}"
                items.append({"source": src["id"], "url": url, "lang": "en", "title": f"Certificate issued for {name}",
                              "text": f"Issuer: {row.get('issuer_name', '')}", "domain": name.lstrip("*."),
                              "published": calendar.timegm(time.strptime(row.get("entry_timestamp", "1970-01-01T00:00:00")[:19],
                                                                         "%Y-%m-%dT%H:%M:%S"))})
    return items, state


def fetch_full_text(item, http):
    if not http.allowed(item["url"]):
        return item
    r = http.get(item["url"], attempts=1)
    if r is not None and r.status_code == 200 and "html" in r.headers.get("content-type", ""):
        body = re.sub(r"(?is)<(script|style|nav|footer|header)[^>]*>.*?</\1>", " ", r.text)
        paras = [clean(p) for p in re.findall(r"(?is)<p[^>]*>(.*?)</p>", body)]
        text = " ".join(p for p in paras if len(p) > 40)
        if len(text) > len(item["text"]):
            item["text"] = text[:6000]
    return item
