# AGENTS.md — Solution 10 (CCTV weapon detection)

Three parts: **agent brief**, **agentic workflow**, **change log**. Repository rules: [`/AGENTS.md`](../../AGENTS.md).

---

## 1. Agent brief

**Purpose:** detect weapons on existing CCTV streams, alert the security desk with evidence, and learn from
operator verdicts. Runs on CPU on IBM Power.

**Invariants (safety-critical):**

* **No face recognition, no person identification, no tracking of individuals across cameras.** Do not add them.
* **A human verifies every alert.** No automatic lock-down, door control or police dispatch from this code.
* Alerts require: class threshold AND (person overlap when enabled) AND `min_hits` confirmations AND cooldown.
  Any change that loosens these needs a new false-alarm gate run (`eval/false_alarm_rate.py`).
* A new model goes live only after **both gates** (IMPLEMENTATION Steps 7 and 8) pass.
* Event records are hash-chained (`events.py`): never UPDATE protected fields (`id, ts, camera, label, score, box,
  sha256`); reviews are separate fields and audited. Retention only through `EventStore.purge()`.
* Privacy masks are applied before analysis AND storage.
* Exports of RT-DETR(v2) must use `common/python/vision/export_rtdetr.py` (dynamo exporter + parity check; the
  TorchScript exporter silently produces wrong scores for RT-DETRv2).
* Camera URLs and credentials only via environment variables; tokens only as SHA-256 in `tokens.yaml`.

**Map:**

| Change | File |
|---|---|
| Alert rules, thresholds, cameras | `config.yaml`, `src/pipeline.py: process()` |
| Storage, chain, retention | `src/events.py` |
| Notifications | `src/notify.py` |
| API / console | `src/app.py`, `static/console.html` |
| Video, detector, tracker, auth (shared with Solution 12) | `common/python/vision/` |

**Verify:**

```bash
python -m pytest -q tests                                                     # 8 passed
python ../../common/python/vision/export_rtdetr.py --model PekingU/rtdetr_v2_r18vd --out /tmp/coco   # parity OK
python eval/false_alarm_rate.py --negatives <normal footage> --max-per-camera-hour 0.02            # gate
```

---

## 2. Agentic workflow (runtime)

The detector raises alerts; an **alert triage agent** helps the operator decide faster, without ever deciding
for them.

```
 new event (snapshot, clip, camera, label, score)
        │
        ▼
 [Triage agent]  Granite 4.2-8B on llama-server, thinking on, max 4 tool calls, READ-ONLY tools
   tools:
     camera_context(camera_id)        -> location, zone type (entrance, armoury, guard post), schedule
     recent_events(camera_id, hours)  -> earlier alerts / false alarms on this camera (patterns: "poster on wall")
     authorised_presence(zone, time)  -> is armed guard presence expected here now? (from the guard roster)
     similar_false_alarms(event_id)   -> past false alarms with a similar box position/size on this camera
        │
        ▼
 triage note in Arabic + English: "Armed guard post, shift change at 08:00: likely authorised"
                                 or "Entrance lobby, no armed staff scheduled: treat as real"
        │
        ▼
 operator console shows the note next to the snapshot ── the operator confirms or marks a false alarm
```

**Guardrails:** the agent has no write tools and cannot suppress alerts; its note is advisory and labelled as
AI-generated; it never receives video of people beyond the event snapshot; every note is stored with the event.
A text model cannot see the image, so the note relies only on context tools, never on "looking" at the snapshot.

**Implementation path:** Solution 09 `src/agent.py` (tool loop + audit), tools as thin functions over
`events.db`, the camera inventory and the guard roster.

---

## 3. Agent change protocol

1. Change, run every **Verify** command; for detector/rule changes, record both gate results.
2. Docs changed → `python tools/build_docx.py solutions/10-cctv-weapon-detection`.
3. Append a row below.

### Agent change log

| Date | Agent / author | Change | Verification result |
|---|---|---|---|
| 2026-11-16 | Claude Code (Opus 5.5) | Initial version: RTSP readers, motion gate, RT-DETRv2 ONNX detector, alert rules, tracking, evidence (hash chain, clips), notifications, escalation, camera-offline alerts, operator console + API with roles, retention, training/export/eval/benchmark tools, release gates | x86 CPU: `pytest` 8 passed; RT-DETRv2 export parity 1.8e-5 with the dynamo exporter (TorchScript exporter found wrong by ~0.1 and replaced); end-to-end false-alarm run on a 30 s synthetic video with the COCO baseline: 0 alerts, 3.4 s; benchmark 108 ms/frame at 8 threads. No weapon model trained (needs site data); not yet run on Power hardware |
| 2026-11-17 | Claude Code (Opus 5.5) | Zero-shot backend (OWLv2) with its own thresholds/fps; window confirmation mode for low fps; OWLv2 square-padding box fix; file replay mapped to wall-clock time; fractional clip frame rate; console fetches each snapshot once; `.webm` support in eval; `finetune/autolabel.py` teacher | Full pipeline on 8 Wikimedia Commons videos: false alarms 0 (5.6 min, unchanged), weapon videos caught 1/5 → 3/5 after window confirmation; live service: alert, clip, confirm, audit, integrity OK; audit spam 12 rows/40 s → 1 per view; clip duration 7 s → 14 s (correct); `pytest` 9 passed |
