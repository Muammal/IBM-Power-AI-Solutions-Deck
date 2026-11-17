# Solution 12 — Traffic cameras: plate reading, section-speed enforcement, traffic analytics

<div dir="rtl">

**ملخص:** نظام لكاميرات الطرق يقرأ لوحات المركبات (ANPR)، ويحسب **السرعة المتوسطة** بين نقطتين على الطريق
(أدق طريقة لإثبات المخالفة)، ويُنشئ مخالفة موقّعة رقمياً لا تُعتمد إلا بعد مراجعة موظف مختص، مع إحصاءات مرورية
مجهولة الهوية. القراءات التي لا تتحول إلى مخالفة تُحذف تلقائياً بعد 24 ساعة. يعمل على المعالج المركزي لخوادم
IBM Power في مركز بيانات مديرية المرور.

</div>

## The problem

Speeding on Iraqi highways (Baghdad–Basra, Baghdad–Erbil, Kirkuk–Sulaymaniyah ...) causes thousands of deaths a
year. Spot radar is easy to brake for; average-speed (section) control changes behaviour over whole stretches.
Traffic authorities also need counts and travel times to plan roads. The data (plates, locations, times) is
sensitive and must stay in the authority's data center.

## What it does

```
 entry camera (km 12)                                   exit camera (km 17)
   RTSP 10 fps ─► RT-DETRv2 (vehicles + plates, ONNX) ─► IoU tracker ─► trigger line per lane
                                                          │ crossing time interpolated between frames
                                                          ▼ best 3 plate crops (score × sharpness × size)
                                          CRNN-CTC plate OCR (ONNX, ~1 ms) ─► vote ─► normalise (٠-٩ → 0-9)
                                                          │ format rules · ambiguity check · min confidence
                                                          ▼
                                     PLATE READ (time, lane, crop + overview hashes, clock offset)
                                                          │
             at the exit camera: same plate at the entry camera within max_travel_s?
                                                          ▼
     section speed = surveyed distance / Δt ; LOWER BOUND = distance / (Δt + frame + clock uncertainty)
     violation only if lower bound > limit + tolerance, AND chrony offset ≤ 50 ms at both ends
                                                          ▼
     VIOLATION (pending) ── manifest (times, cameras, hashes, speeds) signed with Ed25519
                                                          ▼
     reviewer console (/) ── approve / reject ── evidence package .zip (images, manifest, signature, public key)

 always: 15-minute counts + mean spot speed per lane (homography) · travel times via keyed plate hashes
 retention: reads not in a violation deleted after 24 h; there is NO plate search API
```

## Why section speed (and not spot speed) for fines

Section speed needs two timestamps and one surveyed distance. Timing error (half a frame at each camera + measured
clock offsets) is subtracted in the driver's favour. Over 5 km at 150 km/h (120 s), the uncertainty of ~0.11 s
changes the speed by about 0.1%. Spot speed from one camera (pixel → road homography) is kept for **statistics
only**: its accuracy depends on calibration, lens and vehicle height, and should not support a fine.

**Legal note:** speed enforcement equipment normally needs type approval / metrological certification by the
competent authority. This software implements the measurement and evidence chain; certification of the complete
installation (cameras, time sources, surveyed distances) is the operator's responsibility (IMPLEMENTATION Step 9).

## Models

| Role | Model | Licence |
|---|---|---|
| Vehicle + plate detector | `PekingU/rtdetr_v2_r18vd` fine-tuned on your camera images (classes: car, truck, bus, motorcycle, license_plate) | Apache-2.0 base |
| Plate OCR | CRNN-CTC (~1M params, defined in `src/ocr.py`) trained on your plate crops | yours |

No permissively licensed, production-grade Iraqi plate model exists publicly; both models are trained on the
authority's own data with the scripts provided. Plate formats are configured as regular expressions: the examples in
`config.yaml` must be replaced by the official formats before go-live.

## Production features

| Area | Implemented |
|---|---|
| Measurement | interpolated line crossing, conservative lower bound, tolerance, chrony clock-offset gate |
| Reading quality | multi-crop voting, format validation, digit-confusion repair, ambiguity guard, min confidence |
| Evidence | SHA-256 of every image, Ed25519-signed manifest, signature check in UI/API, evidence .zip |
| Human review | every violation pending until a reviewer approves/rejects; decisions final and audited |
| Privacy | 24 h deletion of non-violation reads, keyed hashes for statistics, no plate search, role-based access |
| Operations | `/health` (cameras, clock offset), Prometheus metrics, retention job, systemd |
| Tests | `tests/test_traffic.py` (10): maths, ambiguity, full read→violation path, clock gate, retention, API |
| OCR (measured) | on 3,000 synthetic training crops: 98.7% exact on 300 held-out; the release gate rejected it at 99.2% precision on confident reads (needs ≥ 99.5%): real plates need real training data |

## Compute and compliance

CPU-only. The detector dominates the load (~100–170 ms per frame measured on an x86 laptop; benchmark on your LPAR).
At 10 fps per camera plan roughly 2 cores per camera lane group, or reduce to 5 fps on slower roads. The OCR model
is negligible. Detector training benefits from a GPU that passes [`COMPLIANCE.md`](../../COMPLIANCE.md).

## Quick start (after common steps C1–C3)

```bash
cd /opt/ibm-power-ai-solutions/solutions/12-traffic-anpr-speed
../../common/scripts/make_venv.sh && source venv/bin/activate
python -m pip install pytest && python -m pytest -q tests      # 10 tests, no camera or model needed
```

## Files

| Path | Purpose |
|---|---|
| `config.yaml` | cameras, lanes, calibration, sections, plate formats, timing, retention |
| `src/pipeline.py` | detection, tracking, crossing, OCR vote, reads, section speed, violations |
| `src/speed.py` | crossing interpolation, section speed + bound, homography, chrony |
| `src/plates.py`, `src/ocr.py` | plate text rules, CRNN model + ONNX reader |
| `src/traffic_store.py` | reads, violations, statistics, journeys, retention |
| `src/app.py`, `static/review.html` | reviewer API and console, evidence export |
| `finetune/train_plate_ocr.py` | OCR training + ONNX export with parity check |
| `eval/eval_ocr.py`, `eval/eval_speed.py` | release gates: OCR exact match, speed vs reference runs |
