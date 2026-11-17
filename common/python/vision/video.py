"""video.py: camera readers that survive real CCTV networks.

Design choices (each one fixes a failure seen in production video systems):
  * one thread per camera, decoding with PyAV (FFmpeg inside IBM's `av` wheel), RTSP over TCP
  * only the NEWEST frame is kept: if inference is slower than the camera, old frames are dropped instead of
    queueing up (no growing latency, no memory leak)
  * frames are sampled at `fps` (e.g. 2-5 fps is enough for detection; decoding still runs at full rate)
  * automatic reconnect with exponential backoff (1 s -> 60 s), status exposed for /health and metrics
  * a short ring buffer of sampled frames is kept for evidence clips (pre-event seconds)
"""
import collections
import logging
import threading
import time

import av
import numpy as np

log = logging.getLogger("vision.video")


class CameraStream(threading.Thread):
    def __init__(self, cam_id, url, fps=2.0, buffer_seconds=6, transport="tcp", open_timeout_s=10, realtime=None):
        super().__init__(name=f"cam-{cam_id}", daemon=True)
        self.cam_id, self.url, self.fps = cam_id, url, float(fps)
        self.transport, self.open_timeout_s = transport, open_timeout_s
        # Files are read as fast as they are consumed (tests, re-processing); live sources in real time.
        self.realtime = (not url.startswith(("/", "./", "file:")) and "://" in url) if realtime is None else realtime
        self._lock = threading.Lock()
        self._latest = None                      # (timestamp, rgb ndarray)
        self._consumed = True
        self.buffer = collections.deque(maxlen=max(1, int(buffer_seconds * self.fps)))
        self._stop = threading.Event()
        self.status = {"camera": cam_id, "connected": False, "last_frame_ts": None, "reconnects": 0,
                       "frames_sampled": 0, "last_error": None, "finished": False}

    def stop(self):
        self._stop.set()

    def latest(self):
        """Newest frame not yet returned, or None. Never blocks."""
        with self._lock:
            if self._consumed or self._latest is None:
                return None
            self._consumed = True
            return self._latest

    def recent_frames(self):
        with self._lock:
            return list(self.buffer)

    def _open(self):
        opts = {"rtsp_transport": self.transport, "timeout": str(int(self.open_timeout_s * 1e6))} \
            if self.url.startswith("rtsp") else {}
        return av.open(self.url, options=opts, timeout=self.open_timeout_s)

    def run(self):
        backoff = 1.0
        while not self._stop.is_set():
            try:
                with self._open() as container:
                    stream = container.streams.video[0]
                    stream.thread_type = "AUTO"
                    self.status.update(connected=True, last_error=None)
                    backoff = 1.0
                    next_t = 0.0
                    for frame in container.decode(stream):
                        if self._stop.is_set():
                            return
                        if self.realtime:
                            now = time.time()
                        else:  # file: use the media clock, so tests behave like a live camera
                            now = float(frame.pts * stream.time_base) if frame.pts is not None else time.time()
                        if now < next_t:
                            continue
                        next_t = now + 1.0 / self.fps
                        rgb = frame.to_ndarray(format="rgb24")
                        if not self.realtime:  # wait until the consumer took the previous frame
                            while not self._consumed and not self._stop.is_set():
                                time.sleep(0.002)
                        with self._lock:
                            self._latest, self._consumed = (now, rgb), False
                            self.buffer.append((now, rgb))
                        self.status["last_frame_ts"] = now
                        self.status["frames_sampled"] += 1
                if not self.realtime:
                    self.status.update(finished=True, connected=False)
                    return
                raise ConnectionError("stream ended")
            except Exception as exc:  # network drops, camera reboots, bad credentials
                self.status.update(connected=False, last_error=f"{type(exc).__name__}: {exc}"[:200])
                self.status["reconnects"] += 1
                log.warning("camera %s: %s; reconnect in %.0fs", self.cam_id, self.status["last_error"], backoff)
                self._stop.wait(backoff)
                backoff = min(60.0, backoff * 2)


def apply_masks(rgb, polygons):
    """Black out privacy-mask polygons (pixel coordinates) before any analysis or storage."""
    if not polygons:
        return rgb
    import cv2
    out = rgb.copy()
    for poly in polygons:
        cv2.fillPoly(out, [np.asarray(poly, np.int32)], (0, 0, 0))
    return out


def write_clip(frames, path, fps):
    """Write [(ts, rgb), ...] to an H.264 MP4 (evidence clip for the operator)."""
    if not frames:
        return None
    h, w = frames[0][1].shape[:2]
    with av.open(str(path), mode="w") as out:
        st = out.add_stream("libx264" if "libx264" in av.codecs_available else "mpeg4", rate=max(1, int(round(fps))))
        st.width, st.height, st.pix_fmt = w - w % 2, h - h % 2, "yuv420p"
        for _, rgb in frames:
            fr = av.VideoFrame.from_ndarray(rgb[: st.height, : st.width], format="rgb24")
            for pkt in st.encode(fr):
                out.mux(pkt)
        for pkt in st.encode():
            out.mux(pkt)
    return str(path)
