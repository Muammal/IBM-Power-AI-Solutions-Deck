# Solution 11 — Step-by-step implementation

Target: RHEL 9 ppc64le on Power10/Power11, **on-premises** (preferred: alerts and briefings stay internal) or
**IBM Cloud PowerVS**. The LPAR needs outbound HTTPS to the configured public sources only (allow-list them on the
proxy/firewall). Each step ends with a **Check**.

## Step 0 — Purpose and rules

* Written purpose: situational awareness for the safety of staff, operations and IT. Events, not people.
* Source policy: only public sources whose terms allow automated access; respect rate limits; no logins, no
  closed groups, no personal accounts. Legal review of the source list.
* Retention: `retention.items_days`, `retention.events_days`.

**Check:** purpose and source list approved.

## Step 1 — Platform

[`common/README.md`](../../common/README.md) C1–C5 (`medium`, Granite 4.2-8B on :8080). Firewall: allow HTTPS to
the hosts in `config.yaml: sources` (news.un.org, www.aljazeera.net, api.gdeltproject.org, www.cisa.gov, crt.sh).

**Check:** `curl -sI https://news.un.org` works from the LPAR (through your proxy: set `HTTPS_PROXY`).

## Step 2 — Environment and tests

```bash
cd /opt/ibm-power-ai-solutions/solutions/11-osint-situational-awareness
../../common/scripts/make_venv.sh && source venv/bin/activate
python -m pip install pytest && python -m pytest -q tests
```

**Check:** `8 passed`.

## Step 3 — Configure your organisation

* `organisation.domains` and `brand_keywords`: your real domains and brand strings (for lookalike detection).
* `data/inventory.csv`: one row per product in your estate (`vendor,product,version`), e.g. exported from your CMDB.
  Start from `data/inventory.example.csv`.
* `collection.user_agent`: include a real contact address (sources appreciate it, some require it).

**Check:** `python -c "import sys; sys.path.insert(0,'src'); from enrich import lookalike; print(lookalike('<your-brand>-login.com', __import__('app').CFG['organisation']))"` prints a reason.

## Step 4 — Sources

Keep the verified defaults and add Iraqi/Kurdish outlets that publish RSS and allow automated access. For each new
feed: open the URL in a browser, check it is XML with recent items, and add a line under `sources`.
The service reports unreachable or **stale** feeds (newest item older than `max_item_age_days`) in `/health`.

**Check:** first collection (Step 6) shows every source without `last_error`.

## Step 5 — Alert rules and channels

Edit `alerts` (categories, governorates where you operate, minimum severity, brand mentions) and the webhook
URLs in `.env` (`SOC_WEBHOOK_URL`, ...). Point them at a test channel first.

**Check:** rules reviewed with the SOC / operations managers.

## Step 6 — First run

```bash
PYTHONPATH=../../common/python python -m vision.auth new-token analyst1 operator    # add to tokens.yaml
python src/app.py &
sleep 120 && curl -s localhost:8011/health | jq
curl -s -H "Authorization: Bearer <token>" "localhost:8011/api/events?hours=24&min_severity=1" | jq length
```

**Check:** `failing_sources` is empty (or explained), `pending_items` goes down, events appear in the dashboard (`/`).

## Step 7 — Measure extraction quality (release gate)

Have analysts label 200+ real items (copy the format of `data/events_gold.jsonl`). Then:

```bash
python eval/eval_extraction.py data/events_gold.jsonl --min-category 0.8 --min-governorate-f1 0.8
```

**Check:** `GATE PASSED`. If not: adjust category definitions in `src/enrich.py: SYSTEM`, or use Granite 4.2-8B instead of 3B.

## Step 8 — Production

```bash
sudo systemctl enable --now ai-app@11-osint-situational-awareness
```

* Analysts use the dashboard; corrections (`POST /api/events/{id}/correction`) are audited.
* Monitor `osint_items_pending`, `osint_alerts_total` and `/health` in Prometheus/Grafana.
* Switch webhooks from the test channel to the real ones after a week of tuning.

**Check:** daily briefing appears at `briefing.daily_at`; alerts arrive in the channels.

## Step 9 — Continuous improvement (monthly)

```bash
python finetune/build_sft.py
source /opt/venv-finetune/bin/activate        # common/requirements-finetune.txt
python ../../common/python/lora_sft.py --data data/sft_events.jsonl --model ibm-granite/granite-4.2-3b \
    --out out/lora-events --epochs 2 --merge
../../common/scripts/hf_to_gguf.sh out/lora-events/merged /opt/models/granite-4.2-3b-events Q8_0
LLM_URL=http://127.0.0.1:8084/v1 python eval/eval_extraction.py data/events_gold.jsonl   # against the new server
```

**Check:** the fine-tuned model passes the gate with better numbers than the base model.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `failing_sources: [gdelt-iraq]`, HTTP 429 | GDELT rate limit (shared egress IP) | raise `interval_min`; keep ≥ 5 s between requests |
| `feed stale: newest item ...` | the outlet stopped updating its RSS | disable or replace the source |
| `pending_items` keeps growing | llama-server down or too slow | check C5; use 3B for extraction; more cores |
| Duplicate stories | very different headlines | lower `clustering.jaccard` slightly (0.4) |
| Too many alerts | rules too broad | add governorates / raise `min_severity` |
