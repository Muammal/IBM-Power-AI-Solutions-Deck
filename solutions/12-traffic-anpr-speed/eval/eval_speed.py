"""eval_speed.py: compare section speeds measured by the system with reference measurements (release gate).

Before go-live, drive test vehicles through each section at known speeds measured by a reference instrument
(calibrated GNSS speed logger or police lidar) and write data/speed_reference.csv:
    plate,section,reference_kmh,date
Then run, after the test drives were processed:
    python eval/eval_speed.py data/speed_reference.csv --max-overestimate-kmh 0 --max-abs-error-kmh 3
The system must NEVER report a lower-bound speed above the reference speed (no over-estimation against a driver).
"""
import argparse
import csv
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from app import CFG  # noqa: E402
from speed import section_speed  # noqa: E402
from traffic_store import TrafficStore  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reference")
    ap.add_argument("--max-overestimate-kmh", type=float, default=0.0)
    ap.add_argument("--max-abs-error-kmh", type=float, default=3.0)
    a = ap.parse_args()
    store = TrafficStore(ROOT / CFG["storage"]["db"])
    sections = {s["id"]: s for s in CFG["sections"]}
    worst_over, worst_abs, n = -1e9, 0.0, 0
    with open(a.reference, encoding="utf-8") as f:
        for ref in csv.DictReader(f):
            s = sections[ref["section"]]
            ex = store.q("SELECT * FROM reads WHERE plate=? AND camera=? ORDER BY ts DESC LIMIT 1", (ref["plate"], s["exit"]))
            if not ex:
                print("no exit read for", ref["plate"])
                continue
            en = store.find_entry(ref["plate"], s["entry"], ex[0]["ts"], float(s["max_travel_s"]))
            if not en:
                print("no entry read for", ref["plate"])
                continue
            measured, lower, unc = section_speed(float(s["distance_m"]), en["ts"], ex[0]["ts"], en["frame_dt"],
                                                 ex[0]["frame_dt"], (en["clock_offset"] or 0, ex[0]["clock_offset"] or 0))
            r = float(ref["reference_kmh"])
            worst_over, worst_abs, n = max(worst_over, lower - r), max(worst_abs, abs(measured - r)), n + 1
            print(json.dumps({"plate": ref["plate"], "reference": r, "measured": round(measured, 2), "lower_bound": round(lower, 2)}))
    ok = n > 0 and worst_over <= a.max_overestimate_kmh and worst_abs <= a.max_abs_error_kmh
    print(f"runs={n} worst(lower_bound - reference)={worst_over:.2f} km/h worst|measured - reference|={worst_abs:.2f} km/h")
    print("GATE PASSED" if ok else "GATE FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
