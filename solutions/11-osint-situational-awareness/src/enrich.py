"""enrich.py: language, event extraction (Granite 4.2, JSON schema), governorates, story clustering, cyber checks."""
import csv
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1] / "common" / "python"))
from llm_client import LLM  # noqa: E402

from gazetteer import NAMES, find_governorates  # noqa: E402

CATEGORIES = ["security_incident", "protest_civil_unrest", "infrastructure_outage", "weather_environment", "health",
              "economy_markets", "government_policy", "transport", "cyber", "humanitarian", "other"]
EVENT_SCHEMA = {
    "type": "object",
    "properties": {
        "is_event": {"type": "boolean", "description": "true if the item reports a concrete event or development"},
        "category": {"type": "string", "enum": CATEGORIES},
        "governorates": {"type": "array", "items": {"type": "string", "enum": NAMES + ["Nationwide", "Outside Iraq"]}},
        "location_text": {"type": "string"},
        "severity": {"type": "integer", "minimum": 1, "maximum": 5},
        "organizations": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
        "summary_ar": {"type": "string"},
        "summary_en": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["is_event", "category", "governorates", "severity", "summary_ar", "summary_en", "confidence"],
}
SYSTEM = """You are an analyst producing situational-awareness records for an organisation operating in Iraq.
From one public news item, extract the event. Rules:
- Severity 1 = routine, 3 = significant disruption or risk to operations or public safety, 5 = mass-casualty or
  national-scale crisis.
- organizations: government bodies, companies, armed or political groups as ORGANISATIONS. Never list private
  individuals; do not name people at all.
- Summaries: one neutral sentence each in Arabic and English. Do not speculate beyond the text.
- governorates: only Iraqi governorates actually affected; "Nationwide" for national scope."""

_KU = re.compile("[ڕڵۆێەڤ]")            # letters used in Kurdish (Sorani) but not in Arabic
_AR = re.compile("[؀-ۿ]")


def detect_lang(text):
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return "en"
    if sum(bool(_AR.match(c)) for c in letters) / len(letters) < 0.3:
        return "en"
    return "ckb" if len(_KU.findall(text)) >= 2 else "ar"


def extract_event(item, llm):
    text = f"Title: {item['title']}\nText: {item['text'][:3000]}\nSource: {item['source']}"
    ev = llm.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}],
                  json_schema=EVENT_SCHEMA, temperature=0.0, max_tokens=700)
    # deterministic governorate matching wins over / adds to the model's list
    rule_govs = find_governorates(item["title"] + " " + item["text"])
    ev["governorates"] = sorted(set(g for g in ev.get("governorates", []) if g in NAMES + ["Nationwide", "Outside Iraq"])
                                | set(rule_govs))
    ev["organizations"] = [o for o in ev.get("organizations", []) if len(o) < 120][:6]
    return ev


# ------------------------------------------------------------ clustering
_AR_NORM = [("[إأآٱ]", "ا"), ("ة", "ه"), ("ى", "ي"), ("[ً-ٟـ]", "")]
_STOP = {"the", "a", "an", "of", "in", "on", "and", "to", "for", "at", "is", "with", "after", "amid", "over",
         "في", "من", "علي", "الي", "عن", "مع", "و", "بسبب", "بعد", "خلال", "حول", "اثر", "وسط", "هذا", "هذه", "التي", "الذي"}
# Arabic clitics glued to words: "والكهرباء", "بالبصرة", "للكهرباء" must match "الكهرباء" / "البصرة".
_PREFIXES = ("وال", "بال", "كال", "فال", "لل", "ال")


def _stem(w):
    if re.match(r"[؀-ۿ]", w):
        for p in _PREFIXES:
            if w.startswith(p) and len(w) - len(p) >= 3:
                return w[len(p):]
        if w[0] in "وب" and len(w) > 4:
            return w[1:]
    return w


def shingles(title):
    """Normalised word set of a title. Word sets (not bigrams) survive the paraphrasing between outlets."""
    t = title.lower()
    for a, b in _AR_NORM:
        t = re.sub(a, b, t)
    return {_stem(w) for w in re.findall(r"\w+", t) if w not in _STOP and len(w) > 1} - _STOP


def jaccard(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0


# ------------------------------------------------------------ cyber
_HOMO = str.maketrans({"0": "o", "1": "l", "3": "e", "5": "s", "7": "t", "@": "a", "$": "s", "!": "i", "|": "l"})


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def lookalike(domain, org):
    """Returns a reason string if `domain` imitates one of the organisation's domains, else None."""
    domain = domain.lower().strip(".")
    own = [d.lower() for d in org.get("domains", [])]
    if any(domain == d or domain.endswith("." + d) for d in own):
        return None                                                     # our own (sub)domain
    label = domain.split(".")[0].translate(_HOMO).replace("-", "")
    for d in own:
        base = d.split(".")[0]
        if label == base:
            return f"same name as {d} under another domain"
        if lev(label, base) <= 2 and len(base) >= 5:
            return f"edit distance {lev(label, base)} from {d}"
    for kw in org.get("brand_keywords", []):
        if kw.lower() in label:
            return f"contains brand keyword '{kw}'"
    return None


def load_inventory(path):
    p = ROOT / path
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        return [{k: (v or "").strip().lower() for k, v in r.items()} for r in csv.DictReader(f)]


def kev_matches(kev, inventory):
    v, prod = kev["vendor"].lower(), kev["product"].lower()
    return [i for i in inventory if i.get("vendor") and i["vendor"] in v and i.get("product") and
            (i["product"] in prod or prod in i["product"])]


def new_llm(cfg):
    return LLM(cfg["enrichment"]["llm_url"])
