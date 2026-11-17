# Solution 10 — Weapon detection across an organisation's CCTV cameras

<div dir="rtl">

**ملخص:** خدمة تحلّل بث كاميرات المراقبة الحالية (RTSP) في المصارف والوزارات والجامعات والمنشآت النفطية، وتكتشف
ظهور سلاح (مسدس، بندقية، سكين) بيد شخص، وترسل تنبيهاً فورياً لغرفة الأمن مع صورة ومقطع فيديو قصير. القرار النهائي
دائماً لموظف الأمن (تأكيد / إنذار كاذب)، وكل قرار يُستخدم لتحسين النموذج. تعمل على المعالج المركزي لخوادم IBM Power
داخل المنشأة، ولا تستخدم التعرف على الوجوه.

</div>

## The problem

Organisations in Iraq run hundreds of CCTV cameras watched by a few guards. A weapon at an entrance is visible
on camera seconds before anything happens, but nobody is looking at that screen at that moment. The footage is
sensitive and must stay inside the organisation.

## What it does

```
 IP cameras / NVR (RTSP sub-stream, 720p)            one LPAR, CPU only
        │
        ▼
 CameraStream (PyAV, reconnect, newest-frame-only, 6 s ring buffer)  ──► camera offline alert
        │
 privacy masks ──► motion gate (skips static frames: most of the day)
        │
        ▼
 RT-DETRv2 detector, ONNX Runtime (Power10 MMA), YOUR fine-tuned classes: person, handgun, rifle, knife
        │
 rules: per-class threshold · weapon must touch a person · seen in ≥ 3 consecutive analysed frames · cooldown
        │
        ▼
 EVENT: snapshot (raw + annotated, SHA-256) · 10 s MP4 clip · hash-chained record · audit log
        │                                    │
        ├──► webhook to VMS / PSIM / SOC     ├──► e-mail with snapshot
        ▼
 operator console (/): confirm or false alarm  ──► escalation if nobody reviews in 2 min
        │
        └──► reviewed events ──► finetune/build_dataset.py ──► retrain ──► release gates ──► new model
```

No face recognition, no identification of people: the system detects objects and leaves identification and
response to the security team.

## Production features

| Area | What is implemented |
|---|---|
| Video | RTSP over TCP, auto-reconnect with backoff, frame dropping instead of queueing, privacy masks |
| Scale | motion gating, per-camera analysis fps, measured capacity (`benchmark.py`) |
| False-alarm control | class thresholds, person-overlap rule, multi-frame confirmation, cooldown, hard-negative retraining |
| Alerting | webhooks + e-mail with retries in a background queue, escalation, camera-offline alerts |
| Evidence | raw + annotated snapshot, MP4 clip, SHA-256, tamper-evident hash chain (`/api/integrity`) |
| Security | bearer tokens with roles (hashed at rest), audit of every view/review, localhost binding |
| Operations | `/health`, Prometheus `/metrics`, systemd, retention purge, release gates for new models |
| Tests | `tests/test_weapons.py` (rules, chain, retention, escalation, auth, review) |

## Models

| Role | Model | Licence |
|---|---|---|
| Base detector | `PekingU/rtdetr_v2_r18vd` (RT-DETRv2, 20M params, COCO) | Apache-2.0 |
| Production detector | the base fine-tuned on **your** labelled footage: person, handgun, rifle, knife | yours |

There is no trustworthy, permissively licensed, off-the-shelf weapon model: the few public checkpoints are small,
unvetted, or carry unclear licences. A production deployment therefore **trains its own detector** on footage from its
own cameras (Steps 5–7), and the system will not start with a model that lacks the configured weapon classes.

## Compute and compliance

CPU-only inference on Power10/Power11. Measured on an x86 laptop (Power numbers will differ: run
`benchmark.py` on the LPAR): 108 ms per 640×640 frame with 8 threads → about 15 cameras at 2 analysed fps with 30%
motion. Plan one `medium` LPAR (8 cores) per ~10–15 cameras until you have your own benchmark.
Detector **training** is GPU-friendly; a GPU must pass [`COMPLIANCE.md`](../../COMPLIANCE.md) (T4/V100/A10 class).
Legal: CCTV analytics needs a lawful basis and signage under your organisation's policy; see `IMPLEMENTATION.md` Step 0.

## Quick start (after common steps C1–C3)

```bash
cd /opt/ibm-power-ai-solutions/solutions/10-cctv-weapon-detection
../../common/scripts/make_venv.sh && source venv/bin/activate
python -m pytest -q tests                      # 8 tests, no model or camera needed
```

The full path from cameras to a trained, gated model is in [`IMPLEMENTATION.md`](IMPLEMENTATION.md).

## Files

| Path | Purpose |
|---|---|
| `config.yaml` | cameras, thresholds, rules, notification, retention |
| `src/app.py` | service entry point: API, console, wiring |
| `src/pipeline.py` | analysis engine (motion, detection, rules, tracking, alerts, clips, escalation) |
| `src/events.py` | event store, hash chain, reviews, retention |
| `src/notify.py` | webhook / e-mail delivery with retries |
| `static/console.html` | operator console (Arabic/English) |
| `finetune/build_dataset.py`, `merge_coco.py` | reviews → training data |
| `eval/false_alarm_rate.py` | full-pipeline false alarms per camera-hour + staged recall (release gate) |
| `../../common/python/vision/` | shared video, detector, tracker, evidence, auth, training, export, eval, benchmark |
