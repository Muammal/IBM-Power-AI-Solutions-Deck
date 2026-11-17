"""motion.py: a cheap motion gate. Running the detector costs ~100-400 ms per frame on CPU; computing motion on a
96x54 grey thumbnail costs well under 1 ms. Most CCTV scenes are static most of the time, so gating the detector
on motion multiplies the number of cameras one LPAR can watch.
"""
import numpy as np


class MotionGate:
    def __init__(self, min_changed_fraction=0.004, threshold=18, learning_rate=0.05, keepalive_s=10.0):
        self.min_frac, self.threshold, self.alpha = min_changed_fraction, threshold, learning_rate
        self.keepalive_s = keepalive_s          # run the detector at least this often even without motion
        self.background = None
        self.last_run = -1e18

    @staticmethod
    def _thumb(rgb):
        h, w = rgb.shape[:2]
        sy, sx = max(1, h // 54), max(1, w // 96)
        return rgb[::sy, ::sx].mean(axis=2).astype(np.float32)

    def check(self, rgb, ts):
        """True when the frame should go to the detector."""
        g = self._thumb(rgb)
        if self.background is None or self.background.shape != g.shape:
            self.background = g
            self.last_run = ts
            return True
        changed = float((np.abs(g - self.background) > self.threshold).mean())
        self.background += self.alpha * (g - self.background)
        if changed >= self.min_frac or ts - self.last_run >= self.keepalive_s:
            self.last_run = ts
            return True
        return False
