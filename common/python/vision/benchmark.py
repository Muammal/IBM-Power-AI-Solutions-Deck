"""benchmark.py: measure detector speed on THIS machine and turn it into a camera capacity estimate.

    python common/python/vision/benchmark.py --model models/detector/model.onnx --fps 2 --motion-duty 0.3
Run it on the target LPAR (Power10/Power11, SMT4) with the real model before promising a camera count.
"""
import argparse
import os
import time

import numpy as np
import onnxruntime as ort


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--fps", type=float, default=2.0, help="analysed frames per second per camera while active")
    ap.add_argument("--motion-duty", type=float, default=0.3, help="share of time a camera sees motion (measure it)")
    ap.add_argument("--threads", default="", help="comma list, default: 4,8,all")
    a = ap.parse_args()
    n_cpu = os.cpu_count() or 4
    thread_list = [int(t) for t in a.threads.split(",")] if a.threads else sorted({4, 8, n_cpu})
    print(f"{'threads':>7} {'batch':>5} {'ms/frame':>9} {'frames/s':>9} {'cameras':>8}")
    for th in thread_list:
        so = ort.SessionOptions()
        so.intra_op_num_threads = th
        s = ort.InferenceSession(a.model, so, providers=["CPUExecutionProvider"])
        for b in (1,):  # exported graphs are batch 1 (see export_rtdetr.py)
            x = np.random.rand(b, 3, a.size, a.size).astype(np.float32)
            s.run(None, {"pixel_values": x})
            n, t = 5, time.perf_counter()
            for _ in range(n):
                s.run(None, {"pixel_values": x})
            ms = (time.perf_counter() - t) / n / b * 1000
            fps_total = 1000 / ms
            cams = fps_total / (a.fps * a.motion_duty)
            print(f"{th:>7} {b:>5} {ms:>9.0f} {fps_total:>9.1f} {cams:>8.0f}")
    print("cameras = frames/s / (fps x motion duty). Keep 30% headroom for alert bursts and clip encoding.")


if __name__ == "__main__":
    main()
