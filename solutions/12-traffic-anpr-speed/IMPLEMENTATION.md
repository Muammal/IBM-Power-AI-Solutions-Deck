# Solution 12 — Step-by-step implementation

Target: RHEL 9 ppc64le on Power10/Power11. **On-premises** in the traffic authority's data center (cameras reach
it over the roadside network / MPLS). IBM Cloud PowerVS can host training and test environments with recorded
footage. Each step ends with a **Check**.

## Step 0 — Legal basis and procedure

* Legal basis for automated enforcement and for processing plates; who signs notices.
* Retention: non-violation reads (default 24 h), violations (per law), evidence export procedure.
* Reviewer roles and the rule that **no notice is issued without human review**.
* Type approval / metrological certification requirements for the installation (Step 9).

**Check:** documented legal basis, retention periods and review procedure.

## Step 1 — Platform and time

[`common/README.md`](../../common/README.md) C1–C3. Then accurate time, which the speed calculation depends on:

```bash
sudo dnf install -y chrony && sudo systemctl enable --now chronyd
# point /etc/chrony.conf at your stratum-1 / GPS time servers, then:
chronyc tracking | grep "System time"
```

**Check:** `System time` offset below 1 ms; `common/scripts/check_power.sh` → `RESULT: PASS`.

## Step 2 — Environment and tests

```bash
cd /opt/ibm-power-ai-solutions/solutions/12-traffic-anpr-speed
../../common/scripts/make_venv.sh && source venv/bin/activate
python -m pip install pytest && python -m pytest -q tests
```

**Check:** `10 passed`.

## Step 3 — Cameras and geometry

1. Camera per direction, plate-readable resolution (plate ≥ 120 px wide), IR illumination for night, shutter fast
   enough for 120+ km/h (1/1000 s or faster), 10 fps stream.
2. In `config.yaml` per camera: one trigger `line` per lane (pixels, across the lane at a point where plates are
   sharp). Optional `ground_points` (4 image points ↔ metres on the road) for spot-speed statistics.
3. Per section: entry/exit camera, **surveyed** distance between the two trigger lines (by a licensed surveyor),
   limit, tolerance agreed with the authority.

**Check:** a still frame from each camera with the lines drawn in (`cv2.line`) reviewed by the traffic engineer.

## Step 4 — Collect data

Run the cameras through the recorder (your VMS) for 2–4 weeks covering day, night, rain, dust. Export:
* 3,000+ frames with boxes for `car`, `truck`, `bus`, `motorcycle`, `license_plate` (COCO, CVAT).
* 20,000+ plate crops with their text (`labels.csv`: `file,text`), including Kurdistan Region plates, old and new
  designs, and damaged/dirty plates.

**Check:** datasets split by day (not by frame) into train/val/test.

## Step 5 — Train and gate the detector

```bash
source /opt/venv-vision-train/bin/activate     # common/requirements-vision-train.txt
python ../../common/python/vision/train_rtdetr.py --train data/det/train --val data/det/val --out out/detector --epochs 40
python ../../common/python/vision/export_rtdetr.py --model out/detector/best --out models/detector --image data/det/test/images/<x>.jpg
python ../../common/python/vision/eval_detector.py --model models/detector/model.onnx --data data/det/test \
    --thresholds '{"license_plate":0.35,"car":0.35}' --gate '{"license_plate":{"recall":0.97}}'
```

**Check:** `parity ... OK` and `GATE PASSED` (plates found in ≥ 97% of test vehicles).

## Step 6 — Plate formats and OCR

1. Replace `ocr.formats` with the **official** plate formats (federal and Kurdistan Region; all series in use).
2. Train and gate the OCR:

```bash
python finetune/train_plate_ocr.py --data data/ocr/train --val data/ocr/val --epochs 60
python eval/eval_ocr.py data/ocr/test --min-exact 0.97 --min-precision-confident 0.995
```

`precision_when_confident` is what protects drivers: among reads the system would use for a violation, at least
99.5% must be exactly right. Raise `ocr.min_confidence` until it is.

**Check:** `GATE PASSED`.

## Step 7 — Capacity

```bash
python ../../common/python/vision/benchmark.py --model models/detector/model.onnx --fps 10 --motion-duty 1
```

**Check:** frames/s ≥ cameras × fps with 30% headroom; otherwise add cores, lower fps, or a second LPAR.

## Step 8 — Go live in observation mode

```bash
openssl rand -hex 32 | sed 's/^/PLATE_HMAC_KEY=/' >> .env && chmod 600 .env
PYTHONPATH=../../common/python python -m vision.auth new-token officer.k reviewer   # -> tokens.yaml
sudo systemctl enable --now ai-app@12-traffic-anpr-speed
curl -s localhost:8012/health
```

Back up `data/keys/evidence_ed25519.pem` in the authority's key vault and give the public key to the legal team.
For 4 weeks reviewers inspect every violation but no notices are sent.

**Check:** `/health` shows all cameras up and `clock_offset_s` near 0; violations appear in the console with
`signature: ✔ valid`.

## Step 9 — Measurement validation and certification (release gate)

Drive test vehicles with a calibrated GNSS speed logger through each section at several speeds, then:

```bash
python eval/eval_speed.py data/speed_reference.csv --max-overestimate-kmh 0 --max-abs-error-kmh 3
```

The lower bound must never exceed the reference speed. Hand the results, surveyed distances, chrony logs and
software version to the certifying body.

**Check:** `GATE PASSED`; certification obtained before any notice is issued.

## Step 10 — Operations

* Daily: `/health`, `traffic_clock_offset_seconds` and `traffic_camera_up` in monitoring; alert on clock > 20 ms.
* Weekly: sample 50 approved violations for a second review (quality control).
* Monthly: retrain the OCR with new crops (especially new plate series); both gates before deployment.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| No violations although cars are fast | chrony offset unknown/too large (see logs) | fix NTP; `chronyc tracking` |
| Many `unread` plates | plates too small / motion blur | zoom, faster shutter, IR, move the trigger line |
| Reads but no matches between cameras | different formats or OCR errors at one site | `eval_ocr.py` per site; retrain with that site's crops |
| Ambiguous reads (letter vs digit) | formats overlap | tighten `ocr.formats` to the official ones |
| `410 read expired` in the console | violation older than reads retention for images | reads in violations are kept; check `retention` settings |
