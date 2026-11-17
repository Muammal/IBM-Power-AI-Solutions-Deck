"""speed.py: timing and speed.

Section (average) speed = surveyed distance / time between the two trigger-line crossings. It is the evidential
method because it needs only two accurate timestamps and one surveyed distance, not a calibrated speed sensor.
Every uncertainty (frame interval at each camera, clock offsets) is subtracted in the DRIVER'S FAVOUR: a
violation is raised only when even the lowest plausible speed exceeds limit + tolerance.

Spot speed from one camera (homography to road coordinates) is for traffic statistics only, never for fines.
"""
import re
import subprocess

import numpy as np


def crossing_time(p0, t0, p1, t1, line):
    """If the point moved from p0 (time t0) to p1 (time t1) across the segment `line`, return the interpolated
    crossing time, else None. Points are (x, y) pixels (bottom-centre of the vehicle box)."""
    (ax, ay), (bx, by) = line

    def side(p):
        return (bx - ax) * (p[1] - ay) - (by - ay) * (p[0] - ax)

    s0, s1 = side(p0), side(p1)
    if s0 == 0 or s0 * s1 > 0:
        return None
    f = s0 / (s0 - s1)                                  # fraction of the way from p0 to p1
    x, y = p0[0] + f * (p1[0] - p0[0]), p0[1] + f * (p1[1] - p0[1])
    lo_x, hi_x = sorted((ax, bx))
    if not (lo_x - 5 <= x <= hi_x + 5):                 # crossed the infinite line outside this lane
        return None
    return t0 + f * (t1 - t0)


def section_speed(distance_m, t_entry, t_exit, frame_dt_entry, frame_dt_exit, clock_offsets=(0.0, 0.0)):
    """Returns (measured_kmh, lower_bound_kmh, uncertainty_s)."""
    dt = t_exit - t_entry
    if dt <= 0:
        raise ValueError("exit before entry")
    # interpolation error is at most half a frame interval at each camera; clock offsets add directly
    unc = 0.5 * frame_dt_entry + 0.5 * frame_dt_exit + abs(clock_offsets[0]) + abs(clock_offsets[1])
    return distance_m / dt * 3.6, distance_m / (dt + unc) * 3.6, unc


class Homography:
    def __init__(self, image_pts, world_pts):
        import cv2
        self.H, _ = cv2.findHomography(np.float32(image_pts), np.float32(world_pts))

    def to_world(self, pts):
        p = np.c_[np.float64(pts), np.ones(len(pts))] @ self.H.T
        return p[:, :2] / p[:, 2:3]

    def speed_kmh(self, track_points):
        """track_points: [(t, (x, y)), ...] -> least-squares speed over the track (km/h) or None."""
        if len(track_points) < 4:
            return None
        t = np.array([p[0] for p in track_points])
        w = self.to_world([p[1] for p in track_points])
        if t[-1] - t[0] < 0.3:
            return None
        vx, vy = np.polyfit(t - t[0], w[:, 0], 1)[0], np.polyfit(t - t[0], w[:, 1], 1)[0]
        return float(np.hypot(vx, vy) * 3.6)


def chrony_offset():
    """Current system clock offset from NTP in seconds (chrony), or None if it cannot be read."""
    try:
        out = subprocess.run(["chronyc", "tracking"], capture_output=True, text=True, timeout=5).stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return parse_chrony(out)


def parse_chrony(text):
    m = re.search(r"System time\s*:\s*([0-9.]+) seconds (fast|slow)", text)
    if not m:
        return None
    v = float(m.group(1))
    return v if m.group(2) == "fast" else -v
