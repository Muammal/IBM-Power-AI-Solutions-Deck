# Solution 10 — Step-by-step implementation

Target: RHEL 9 ppc64le on Power10/Power11. **On-premises is the normal deployment** (cameras are on the site
network). IBM Cloud PowerVS is used for the training/evaluation environment or for a central SOC that receives
streams over a private link (Direct Link / VPN). Each step ends with a **Check**.

## Step 0 — Governance (before any camera is connected)

* Written purpose: "detect weapons to protect staff and visitors". No face recognition, no tracking of individuals.
* Legal/compliance sign-off for CCTV analytics, signage at entrances, retention periods (`storage.*`).
* A response procedure: who receives alerts, how they verify (camera live view, radio), what they do next.
* Never act on an alert without human verification: the system can be wrong.

**Check:** signed purpose and response procedure; retention periods agreed.

## Step 1 — Platform

[`common/README.md`](../../common/README.md) C1–C3. On-prem: an LPAR on the security VLAN that can reach the
cameras/NVR on RTSP (TCP 554). Start with `medium` (8 cores / 64 GB) for up to ~10 cameras.

**Check:** `common/scripts/check_power.sh` → `RESULT: PASS`; from the LPAR:
`timeout 10 python -c "import av; av.open('rtsp://...').streams.video[0]"` succeeds for one camera.

## Step 2 — Environment and tests

```bash
cd /opt/ibm-power-ai-solutions/solutions/10-cctv-weapon-detection
../../common/scripts/make_venv.sh && source venv/bin/activate
python -m pip install pytest && python -m pytest -q tests
```

**Check:** `8 passed`.

## Step 3 — Camera inventory and sub-streams

For each camera, use the **sub-stream** (640×360 to 1280×720) rather than the 4K main stream: detection runs at
640×640 anyway, and decoding 4K costs CPU. Add each camera to `config.yaml` with its URL as an environment
variable in `.env` (`CAM_LOBBY_1_URL=rtsp://...`, file mode 600). Draw privacy masks for windows facing homes or
public streets.

**Check:** `python -c` loop over the URLs opens every stream; masks reviewed by the privacy officer.

## Step 4 — Baseline and capacity on THIS LPAR

```bash
source /opt/venv-vision-train/bin/activate      # created from common/requirements-vision-train.txt
python ../../common/python/vision/export_rtdetr.py --model PekingU/rtdetr_v2_r18vd --out models/coco
python ../../common/python/vision/benchmark.py --model models/coco/model.onnx --fps 2 --motion-duty 0.3
```

**Check:** `parity ... OK`, and a cameras-per-LPAR number. Size the LPAR from it (keep 30% headroom).

## Step 5 — Collect and label training data (the most important step)

1. **Positives**: staged recordings with **replica/training weapons** (coordinate with security), on your own
   cameras, at every entrance, day and night (IR), different distances and how people really carry them. Aim for
   2,000+ labelled frames per weapon class.
2. **Negatives**: frames with phones, umbrellas, tools, radios, cameras, black bags, and staff uniforms. These
   cut false alarms.
3. Label in **CVAT** or **Label Studio** with boxes for `person`, `handgun`, `rifle`, `knife`; export **COCO 1.0**.
4. Split by **recording session** (not by frame) into `data/train`, `data/val`, `data/test`.

Public datasets can supplement your data only after checking that their licence allows your use.

**Check:** `data/{train,val,test}/annotations.json` exist; every class has ≥ 300 boxes in `test`.

## Step 6 — Train (CPU or a compliant GPU)

```bash
source /opt/venv-vision-train/bin/activate
python ../../common/python/vision/train_rtdetr.py --train data/train --val data/val \
    --out out/detector --base PekingU/rtdetr_v2_r18vd --epochs 40 --size 640
```

CPU (Power10, 16 cores): plan about a day for 2,000 images × 40 epochs; `--size 512 --freeze-backbone` is ~3×
faster. A GPU that passes [`COMPLIANCE.md`](../../COMPLIANCE.md): add `--gpu`, then copy `out/detector/best` back.

**Check:** training ends with `saved out/detector/best`; validation loss went down.

## Step 7 — Export, evaluate, set thresholds (release gate 1)

```bash
python ../../common/python/vision/export_rtdetr.py --model out/detector/best --out models/candidate --image data/test/images/<any>.jpg
python ../../common/python/vision/eval_detector.py --model models/candidate/model.onnx --data data/test \
    --thresholds '{"handgun":0.6,"rifle":0.6,"knife":0.7}' \
    --gate '{"handgun":{"recall":0.85,"precision":0.9},"rifle":{"recall":0.85,"precision":0.9}}'
```

Tune the thresholds until the gate passes, then copy them into `config.yaml: alerting.classes`.

**Check:** `GATE PASSED`.

## Step 8 — Full-pipeline false-alarm test (release gate 2)

Record at least **50 camera-hours of normal activity** (no weapons) and the staged weapon sessions you did NOT use
for training:

```bash
cp -r models/candidate models/detector
python eval/false_alarm_rate.py --negatives /data/recordings/normal --positives /data/recordings/staged_holdout \
    --max-per-camera-hour 0.02 --min-recall 0.9
```

0.02 false alarms per camera-hour means about 1 per camera every 2 days: agree the number with the security
manager. Lower it with: higher thresholds, `min_hits: 4`, more hard negatives (Step 5), masks over problem areas.

**Check:** `GATE PASSED`.

## Step 9 — Operator access

```bash
PYTHONPATH=../../common/python python -m vision.auth new-token sara operator     # one per person
cp tokens.yaml.example tokens.yaml   # paste the printed entries, chmod 600
```

Publish `http://127.0.0.1:8010/` through the security-network reverse proxy with TLS (and SSO if available).

**Check:** a viewer token can see alerts but gets HTTP 403 on review; an operator token can review.

## Step 10 — Go live (shadow first)

1. Configure webhooks/e-mail in `.env`; first point them to a test channel.
2. `sudo systemctl enable --now ai-app@10-cctv-weapon-detection`
3. **Shadow mode for 2 weeks**: operators review every alert in the console, but no response procedure is
   triggered. Track false alarms per camera-hour and response times.
4. Switch the webhooks to the real security-desk channel.

**Check:** `/health` shows all cameras up; `cctv_alerts_total`, `cctv_camera_up` visible in Prometheus/Grafana;
shadow-mode false-alarm rate within the agreed limit.

## Step 11 — Continuous improvement (monthly)

```bash
python finetune/build_dataset.py --classes-from data/train/annotations.json
# correct boxes of confirmed events in CVAT, then:
python finetune/merge_coco.py data/train data/training/from_reviews data/train_v2
# Step 6 with --train data/train_v2, then Steps 7-8 (both gates) before replacing models/detector
```

**Check:** both gates pass and the new model is not worse than the current one on the same test data.

## Operations

* `GET /api/integrity` (admin) verifies the event hash chain; run it daily from monitoring.
* Back up `data/events.db`, `data/media/` and `data/keys/` (the evidence key) with your backup system.
* Alert on `cctv_camera_up == 0` and on missing `/health` in your monitoring.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Camera keeps reconnecting | wrong URL/credentials, UDP blocked | test with `ffprobe`; RTSP over TCP is default |
| High CPU, delayed alerts | too many cameras / main stream | sub-streams, lower `fps`, motion gate on, more cores |
| Repeated false alarms on one camera | object in the scene (poster, statue) | privacy mask or hard negatives from that camera |
| Model refuses to start (`model has no classes`) | COCO baseline still configured | Steps 5–7 |
| No clip for an event | alert at the end of the stream | clips need `clip_post_s` of further frames |
