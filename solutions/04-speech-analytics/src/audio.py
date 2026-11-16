"""audio.py: decode any audio/video file (wav, mp3, m4a, ogg/opus from WhatsApp, mp4...) to the format
Whisper expects: 16 kHz, mono, 16-bit PCM WAV. Uses PyAV (FFmpeg libraries inside IBM's `av` wheel),
so no separate ffmpeg install is needed.
"""
import io
import wave

import av
import numpy as np


def to_pcm16k(path_or_file):
    """Return (int16 numpy array, seconds)."""
    resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
    chunks = []
    with av.open(path_or_file) as container:
        for frame in container.decode(audio=0):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None):  # flush
            chunks.append(out.to_ndarray().reshape(-1))
    pcm = np.concatenate(chunks).astype(np.int16) if chunks else np.zeros(0, np.int16)
    return pcm, len(pcm) / 16000


def to_wav_bytes(pcm):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def to_float32(pcm):
    """For Hugging Face feature extractors (fine-tuning): float32 in [-1, 1]."""
    return pcm.astype(np.float32) / 32768.0
