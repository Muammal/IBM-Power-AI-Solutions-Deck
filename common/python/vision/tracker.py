"""tracker.py: greedy IoU tracker. Good enough for fixed CCTV cameras at 2-10 fps and cheap on CPU.

A track becomes `confirmed` after `min_hits` matched detections; it is dropped after `max_misses` frames without a
match. Alerting only on confirmed tracks removes most single-frame false positives (a phone or a dark umbrella
detected once as a handgun).
"""
import dataclasses
import itertools

from .detector import iou

_ids = itertools.count(1)


@dataclasses.dataclass
class Track:
    label: str
    box: tuple
    score: float
    first_ts: float
    last_ts: float
    hits: int = 1
    misses: int = 0
    id: int = dataclasses.field(default_factory=lambda: next(_ids))
    history: list = dataclasses.field(default_factory=list)   # [(ts, box, score)]
    data: dict = dataclasses.field(default_factory=dict)       # per-track extras (plate crops, alert sent, ...)

    @property
    def center_bottom(self):
        return ((self.box[0] + self.box[2]) / 2, self.box[3])


class IoUTracker:
    def __init__(self, iou_threshold=0.3, min_hits=3, max_misses=4, max_history=60):
        self.iou_threshold, self.min_hits, self.max_misses, self.max_history = iou_threshold, min_hits, max_misses, max_history
        self.tracks = []

    def update(self, detections, ts):
        """Returns (active tracks, tracks that ended this frame)."""
        unmatched = list(detections)
        for t in self.tracks:
            best, best_iou = None, self.iou_threshold
            for d in unmatched:
                if d.label == t.label:
                    v = iou(t.box, d.box)
                    if v >= best_iou:
                        best, best_iou = d, v
            if best is not None:
                unmatched.remove(best)
                t.box, t.score, t.last_ts = best.box, max(t.score, best.score), ts
                t.hits += 1
                t.misses = 0
                t.history.append((ts, best.box, best.score))
                del t.history[:-self.max_history]
            else:
                t.misses += 1
        for d in unmatched:
            self.tracks.append(Track(d.label, d.box, d.score, ts, ts, history=[(ts, d.box, d.score)]))
        ended = [t for t in self.tracks if t.misses > self.max_misses]
        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]
        return self.tracks, ended

    def confirmed(self):
        return [t for t in self.tracks if t.hits >= self.min_hits and t.misses == 0]
