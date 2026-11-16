#!/usr/bin/env bash
# C3 — Create a Python 3.12 venv for one solution and install its pinned requirements
# from IBM's Power-optimized wheel index (pattern from ppc64le/pyeco install_test_example.sh).
#
# Usage (run inside a solution folder):   ../../common/scripts/make_venv.sh [requirements.txt] [venv]
set -euo pipefail

REQ=${1:-requirements.txt}
VENV=${2:-venv}
[[ -f "$REQ" ]] || { echo "No $REQ in $(pwd)"; exit 1; }

[[ -d "$VENV" ]] || python3.12 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel

# --only-binary=:all: -> fail loudly instead of silently compiling a package from source on ppc64le.
# The IBM index URL and --prefer-binary are at the top of every requirements.txt.
# --no-cache keeps the 1 GB+ torch wheel out of ~/.cache/pip.
pip install --no-cache --only-binary=:all: -r "$REQ"

echo
echo "== Installed Power builds (+ppc64le = from IBM's index):"
pip list 2>/dev/null | grep -i ppc64le || true
python - <<'EOF'
import importlib
for mod in ("torch", "transformers", "numpy"):
    try:
        m = importlib.import_module(mod)
        print(f"  {mod:13s} {m.__version__}")
    except ImportError:
        pass
EOF
echo "Done. Activate with:  source $VENV/bin/activate"
