# AGENTS.md — Solution 12 (Traffic ANPR and section speed)

Three parts: **agent brief**, **agentic workflow**, **change log**. Repository rules: [`/AGENTS.md`](../../AGENTS.md).

---

## 1. Agent brief

**Purpose:** read plates, measure section (average) speed with a conservative bound, create signed violations for
human review, and produce anonymous traffic statistics. Runs on CPU on IBM Power.

**Invariants (legal and safety-critical):**

* **Every uncertainty favours the driver.** Violations use `lower_bound_kmh` (distance / (Δt + uncertainty)) against
  limit + tolerance. Never switch to the measured value or remove the uncertainty terms.
* **No violation without valid time:** the chrony offset must be known and ≤ `max_clock_offset_s` at both ends.
* **No violation from uncertain reads:** below `min_confidence`, failing format validation, or ambiguous
  (`is_ambiguous`) reads may feed statistics only.
* **Spot speed is never evidence.** Do not create violations from `spot_kmh`.
* **Human review:** violations are `pending` until a reviewer decides; decisions are final (409 on re-decision) and
  audited; evidence packages only for approved violations.
* Manifests are signed (Ed25519); never change a stored manifest. Hash every image that goes into evidence.
* **Privacy:** no plate search / read-listing endpoints; non-violation reads are deleted after `reads_hours`;
  statistics use keyed hashes (`PLATE_HMAC_KEY`). Tests assert there are no plate-search routes.
* RT-DETR exports only through `common/python/vision/export_rtdetr.py` (dynamo + parity).

**Map:**

| Change | File |
|---|---|
| Geometry, sections, limits, formats, retention | `config.yaml` |
| Measurement maths | `src/speed.py` (tests in `test_traffic.py` must keep passing) |
| Plate text rules | `src/plates.py` |
| Pipeline / violation creation | `src/pipeline.py` |
| Storage / retention | `src/traffic_store.py` |
| Reviewer API / console | `src/app.py`, `static/review.html` |

**Verify:**

```bash
python -m pytest -q tests                                         # 10 passed
python eval/eval_ocr.py data/ocr/test --min-exact 0.97            # OCR gate (needs the trained model)
python eval/eval_speed.py data/speed_reference.csv                # measurement gate (needs reference runs)
```

---

## 2. Agentic workflow (runtime)

A **review assistant agent** prepares each pending violation so the reviewer checks the right things; it can
never approve, reject or contact anyone.

```
 pending violation
        │
        ▼
 [Review assistant] Granite 4.2-8B, thinking on, max 5 tool calls, read-only
   tools:
     violation_facts(id)              -> manifest, signature status, confidences, clock offsets
     read_quality(id)                 -> OCR confidence/agreement at both ends, plate crop sizes, ambiguity flags
     section_health(section, hours)   -> camera uptime, clock offset history, unread rate around that time
     similar_rejections(section)      -> recent rejected violations on this section and their reasons
        │
        ▼
 checklist note: "Both reads 0.98+, same plate format, clocks 2 ms / 3 ms, section healthy → standard review"
            or  "Exit camera unread rate 40% between 02:00-03:00 (IR fault?) → compare plate crops carefully"
        │
        ▼
 reviewer approves / rejects in the console (the agent's note is advisory and labelled as AI-generated)
```

**Guardrails:** no write tools; the agent never sees owner/registration data; notes stored with the violation for audit.

**Implementation path:** Solution 09 `src/agent.py` with tools as SQL queries over `data/traffic.db`.

---

## 3. Agent change protocol

1. Change, run every **Verify** command (both gates for model or measurement changes).
2. Docs changed → `python tools/build_docx.py solutions/12-traffic-anpr-speed`.
3. Append a row below.

### Agent change log

| Date | Agent / author | Change | Verification result |
|---|---|---|---|
| 2026-11-16 | Claude Code (Opus 5.5) | Initial version: detection/tracking/line-crossing pipeline, CRNN-CTC plate OCR (train + ONNX), plate rules with ambiguity guard, section speed with conservative bound and clock gate, signed evidence, reviewer API/console, statistics, retention, gates | x86 CPU: `pytest` 10 passed (incl. full entry→exit→violation path, signature tamper check, clock gate, ambiguity, retention, API roles). A test exposed letter/digit ambiguity between plate formats → `is_ambiguous` guard added. OCR: export parity 2.9e-6; CTC plateau diagnosed (overfit test 63/64 at step 560); synthetic plates (3,000 train / 300 test): 98.7% exact, gate failed on precision-when-confident 99.2% < 99.5% (as designed: raise `min_confidence` or add data). No real Iraqi plate data used; not yet run on Power hardware |
