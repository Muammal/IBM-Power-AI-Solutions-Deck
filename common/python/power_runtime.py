"""power_runtime.py: small helpers every solution uses on IBM Power (and that still work on a laptop).

    from power_runtime import configure_threads, detect_language, load_yaml
    configure_threads()          # call BEFORE importing torch / onnxruntime / xgboost
"""
import os
import re

import yaml

# Arabic script blocks (Arabic, Supplement, Extended-A, Presentation Forms A/B). Kurdish Sorani uses them too.
_ARABIC = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]")


def physical_cores():
    """Cores this process may run on. PyTorch cannot count cores on POWER and would start one thread
    per SMT hardware thread (e.g. 64 threads on 8 cores in SMT8), which slows matrix-heavy work
    because the MMA units are per core. Returns 0 when unknown (not Linux)."""
    try:
        cores = set()
        for cpu in os.sched_getaffinity(0):
            with open(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list") as f:
                cores.add(f.read().strip())
        return len(cores)
    except (AttributeError, OSError):
        return 0


def configure_threads(threads=0):
    """Set the thread count for OpenMP/OpenBLAS (used by IBM's torch wheel), onnxruntime and xgboost.
    Must run before those libraries are imported: they read the variables once."""
    n = int(threads or 0) or physical_cores()
    if n:
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(var, str(n))
    return n


def detect_language(text):
    """'ar' when >= 30% of the letters are Arabic script (Arabic or Kurdish Sorani), else 'en'."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return "en"
    return "ar" if sum(1 for c in letters if _ARABIC.match(c)) / len(letters) >= 0.3 else "en"


def load_yaml(path):
    """Read a UTF-8 YAML file. Environment variables like ${LLM_URL:-http://...} are expanded."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    text = re.sub(r"\$\{(\w+):-([^}]*)\}", lambda m: os.getenv(m.group(1), m.group(2)), text)
    return yaml.safe_load(text)
