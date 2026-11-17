# Solution 11 — National open-source situational awareness and cyber threat intelligence

<div dir="rtl">

**ملخص:** منصة تجمع المعلومات من المصادر العامة المفتوحة (وكالات الأنباء، موجزات الأمم المتحدة، GDELT، كتالوج
الثغرات المستغلة CISA KEV، سجلات شفافية الشهادات) وتحولها إلى أحداث منظمة حسب المحافظة والفئة والخطورة
(انقطاع كهرباء، حوادث أمنية، احتجاجات، فيضانات، أسواق، تهديدات سيبرانية)، وترسل تنبيهات وموجزاً يومياً بالعربية
والإنجليزية لفرق الأمن والعمليات. لا تجمع بيانات عن الأفراد ولا تراقب حسابات التواصل الاجتماعي الشخصية.

</div>

## The problem

Security, operations and business-continuity teams in Iraqi banks, oil companies, telecoms and ministries need to
know quickly what is happening across the country's 19 governorates: an outage in Basra, floods in Duhok, a
security incident on a supply road, a currency-auction rule change, a phishing domain imitating their brand, an
exploited vulnerability in their firewall. That information is public but scattered across sources and languages.

## Scope (by design)

| In scope | Out of scope (not implemented, and must not be added) |
|---|---|
| Public news feeds, UN feeds, GDELT event metadata | Monitoring personal social-media accounts |
| Events: what, where (governorate), how severe | Profiles, dossiers or searches about individuals |
| Organisations as actors (ministries, companies, groups) | Names of private individuals in extracted records |
| Cyber: CISA KEV vs your inventory, lookalike domains of YOUR brand | Collection from closed groups, leaks, or behind logins |

## What it does

```
 public sources (RSS: UN News AR/EN, Al Jazeera AR, + your outlets · GDELT · CISA KEV · crt.sh)
      │  polite collection: identifying User-Agent, ETag/Last-Modified, ≥5 s per host, retries,
      │  robots.txt for full text, stale / failing sources reported in /health
      ▼
 items (title, summary, link; authors dropped)  ──► SQLite (WAL, FTS5 search)
      │
      ▼
 enrichment: language (AR / Kurdish Sorani / EN) · Granite 4.2 JSON-schema event extraction
             (category, governorates, severity 1-5, organisations, AR+EN summary) · gazetteer of Iraq's
             governorates and cities overrides/extends the model · LLM down → items wait, retried
      │
      ├─ cyber: KEV entries matched to data/inventory.csv · CT certificates scored for lookalikes of your domains
      ▼
 story clustering (Arabic-aware word sets, 48 h)  ──► alert rules (category / governorate / severity / brand)
      │                                               ──► webhooks per channel, one alert per story per channel
      ▼
 dashboard (/) · API · daily briefing (factual list + Granite executive summary with citations)
      │
      └─► analyst corrections ──► finetune/build_sft.py ──► LoRA on Granite ──► eval gate
```

## Verified against live sources (2026-11-16)

| Source | Result |
|---|---|
| UN News Arabic + English, Al Jazeera Arabic RSS | collected (31 items in one cycle), enriched, clustered |
| CISA KEV | 1,781 catalogue entries → 87 matched to a 5-product sample inventory (FortiOS, Exchange, Ivanti, ...) |
| INA (Iraqi News Agency) RSS | reachable, but newest item 2026-05-07: detected and reported as **stale**, disabled in config |
| GDELT DOC API | rate-limited this test machine (HTTP 429): reported as a failing source, retried next cycle |

## Models

| Role | Model | Licence |
|---|---|---|
| Event extraction, summaries, briefing | `ibm-granite/granite-4.2-8b` (or -3b) on llama-server | Apache-2.0 |
| Fine-tuned extractor | LoRA on Granite 4.2 with analyst corrections | yours |

## Compute and compliance

CPU-only; the LLM processes a few hundred items per hour on an 8-core LPAR (enrichment queue absorbs bursts).
All data comes from public sources; data minimisation and retention are configured. See [`COMPLIANCE.md`](../../COMPLIANCE.md).

## Quick start (after common steps C1–C5)

```bash
cd /opt/ibm-power-ai-solutions/solutions/11-osint-situational-awareness
../../common/scripts/make_venv.sh && source venv/bin/activate
python -m pip install pytest && python -m pytest -q tests      # 8 tests, offline
cp data/inventory.example.csv data/inventory.csv                # then list YOUR products
python src/app.py                                               # 127.0.0.1:8011
```

## Files

| Path | Purpose |
|---|---|
| `config.yaml` | organisation, sources, rules, channels, briefing, retention |
| `src/collectors.py` | RSS, GDELT, CISA KEV, crt.sh collectors (polite HTTP) |
| `src/gazetteer.py` | Iraq's governorates and cities in Arabic, Kurdish, English |
| `src/enrich.py` | language ID, event schema + extraction, clustering, lookalike domains, KEV matching |
| `src/engine.py` | storage, collect → enrich → cluster → alert → brief → retain |
| `src/app.py`, `static/dashboard.html` | API, scheduler, dashboard |
| `eval/eval_extraction.py`, `data/events_gold.jsonl` | extraction quality gate |
| `finetune/build_sft.py` | analyst corrections → LoRA data |
