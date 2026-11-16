#!/usr/bin/env bash
# Step 03 — Create the Python 3.12 virtual environment and install pinned packages
# from IBM's ppc64le wheel index (pattern taken from ppc64le/pyeco install_test_example.sh).
#
# Usage (from the project root, NOT as root):  ./scripts/03_setup_env.sh
set -euo pipefail
cd "$(dirname "$0")/.."            # always run from the project root

VENV=${VENV:-venv}

if [[ ! -d "$VENV" ]]; then
  python3.12 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

python -m pip install --upgrade pip

# --no-cache keeps the 1+ GB torch wheel out of ~/.cache/pip.
# The IBM index and --prefer-binary are set at the top of requirements.txt, so this
# line installs exactly what the file says. --only-binary=:all: makes pip fail loudly
# instead of silently trying to compile a package from source on ppc64le.
pip install --no-cache --only-binary=:all: -r requirements.txt

# Copy the example .env so HF_TOKEN has a home (never overwrite an existing one).
[[ -f .env ]] || cp .env.example .env 2>/dev/null || true

echo
echo "== Verification"
python -c "import torch, transformers; print(torch.__version__)"
python - <<'EOF'
import torch, transformers, fastapi, yaml
print("transformers", transformers.__version__, "| fastapi", fastapi.__version__)
print("torch threads:", torch.get_num_threads(), "| bf16 supported:",
      torch.ones(2, dtype=torch.bfloat16).sum().item() == 2.0)
EOF
echo
echo "Done. Activate the environment in new shells with:  source $VENV/bin/activate"
