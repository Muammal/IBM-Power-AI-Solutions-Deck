"""false_alarm_rate.py: the metric that decides whether security staff will trust the system.

Runs the FULL pipeline (motion gate, thresholds, person-overlap rule, tracking, cooldown) over recordings that
contain NO weapons (normal days from your own cameras) and reports alerts per camera-hour. Optionally also runs over
staged recordings that DO contain (replica) weapons and reports how many were caught.

    python eval/false_alarm_rate.py --negatives /data/recordings/normal_days --positives /data/recordings/staged
    python eval/false_alarm_rate.py --negatives ... --max-per-camera-hour 0.05 --min-recall 0.9    # release gate
Each staged file must contain a weapon; it counts as caught if at least one alert is raised in that file.
"""
import argparse
import copy
import pathlib
import sys
import tempfile
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from app import CFG, build, effective_config, make_detector  # noqa: E402

VIDEO = {".mp4", ".mkv", ".avi", ".mov", ".ts", ".webm"}


class SilentNotifier:
    def send(self, *a, **k):
        pass


def run_file(path, cfg, detector):
    from vision.video import CameraStream
    cfg = effective_config(cfg)
    cam = CameraStream("eval", str(path), fps=cfg["cameras"][0].get("fps", 2) if cfg["cameras"] else 2,
                       buffer_seconds=6, realtime=False)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:  # Windows: SQLite file may still be open
        c = copy.deepcopy(cfg)
        c["storage"]["db"], c["storage"]["media_dir"] = f"{tmp}/e.db", f"{tmp}/media"
        c["cameras"] = [{"id": "eval", "name": path.name, "fps": cam.fps}]
        c["detector"]["backend"] = "onnx"   # thresholds already applied by effective_config above
        store, pipe = build(c, detector=detector, cameras={"eval": cam}, notifier=SilentNotifier(), start=False)
        cam.start()
        while cam.is_alive() or cam.latest() is not None:
            if pipe.step() == 0:
                time.sleep(0.005)
        while pipe.step():
            pass
        n_alerts = len(store.list(limit=100000))
        media_s = cam.status["media_seconds"] or 0
    return n_alerts, media_s / 3600.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--negatives", required=True)
    ap.add_argument("--positives")
    ap.add_argument("--max-per-camera-hour", type=float)
    ap.add_argument("--min-recall", type=float)
    a = ap.parse_args()
    det = make_detector(effective_config(CFG))     # onnx or zero-shot, as configured
    alerts = hours = 0.0
    for f in sorted(p for p in pathlib.Path(a.negatives).rglob("*") if p.suffix.lower() in VIDEO):
        t0 = time.time()
        n, h = run_file(f, CFG, det)
        alerts, hours = alerts + n, hours + h
        print(f"negative {f.name}: {n} alerts in {h * 60:.1f} min of video ({time.time() - t0:.0f} s to process)", flush=True)
    if hours <= 0:
        print("GATE FAILED: no negative footage was processed (check --negatives)")
        sys.exit(1)
    rate = alerts / hours
    print(f"\nFALSE ALARMS: {alerts:.0f} in {hours:.1f} camera-hours = {rate:.3f} per camera-hour")
    recall = None
    if a.positives:
        files = sorted(p for p in pathlib.Path(a.positives).rglob("*") if p.suffix.lower() in VIDEO)
        caught = 0
        for f in files:
            n, h = run_file(f, CFG, det)
            caught += n > 0
            print(f"positive {f.name}: {n} alerts in {h * 60:.1f} min of video", flush=True)
        recall = caught / max(1, len(files))
        print(f"STAGED WEAPONS: caught {caught}/{len(files)} = {recall:.2%}")
    ok = (a.max_per_camera_hour is None or rate <= a.max_per_camera_hour) and \
         (a.min_recall is None or (recall is not None and recall >= a.min_recall))
    print("GATE PASSED" if ok else "GATE FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
