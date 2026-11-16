#!/usr/bin/env bash
# C2 — Install the OS packages every solution needs on RHEL 9 (ppc64le).
# Package list follows the ppc64le/pyeco Python 3.12 examples (v2026.03.31).
#
# Usage:  sudo common/scripts/os_prep.sh            # packages only
#         sudo common/scripts/os_prep.sh --smt4     # also switch to SMT4 and make it permanent
set -euo pipefail

[[ $EUID -eq 0 ]] || { echo "Run as root: sudo $0"; exit 1; }
. /etc/os-release
case "$ID" in rhel|centos|rocky|almalinux|fedora) ;; *) echo "Unsupported distro: $ID (target: RHEL 9)"; exit 1 ;; esac
[[ "$(uname -m)" == "ppc64le" ]] || echo "WARNING: $(uname -m) is not ppc64le. Continuing anyway."

# python3.12 (+devel)  -> interpreter IBM's wheels are built for (cp312)
# gcc-toolset-13       -> GCC with full -mcpu=power10 / MMA builtin support (llama.cpp, whisper.cpp)
# cmake, git, make     -> building llama.cpp / whisper.cpp
# powerpc-utils        -> ppc64_cpu (SMT control), lparstat
# numactl              -> pin services to one NUMA node on big LPARs
# binutils             -> objdump, used to prove MMA instructions are compiled in
# tesseract (+ara)     -> OCR for Solution 03 (from EPEL on RHEL; skipped if not available)
# libsndfile           -> audio I/O for Solution 04 fine-tuning
# poppler-utils        -> pdftotext / pdftoppm for Solution 03
dnf install -y \
  python3.12 python3.12-devel python3.12-pip \
  gcc gcc-c++ gcc-toolset-13 cmake make git binutils \
  powerpc-utils numactl curl tar unzip jq libsndfile poppler-utils

if ! rpm -q epel-release >/dev/null 2>&1; then
  dnf install -y "https://dl.fedoraproject.org/pub/epel/epel-release-latest-9.noarch.rpm" || \
    echo "NOTE: EPEL not installed (needed only for tesseract in Solution 03)."
fi
dnf install -y tesseract tesseract-langpack-ara tesseract-langpack-eng 2>/dev/null || \
  echo "NOTE: tesseract not installed (only Solution 03 needs it)."

if [[ "${1:-}" == "--smt4" ]]; then
  # SMT4: 4 hardware threads per core. The MMA units are shared per core, so for
  # matrix-heavy AI inference SMT4 usually beats SMT8. Measure with your workload.
  ppc64_cpu --smt=4
  cat >/etc/systemd/system/smt4.service <<'EOF'
[Unit]
Description=Set SMT4 for AI inference
After=multi-user.target
[Service]
Type=oneshot
ExecStart=/usr/sbin/ppc64_cpu --smt=4
[Install]
WantedBy=multi-user.target
EOF
  systemctl daemon-reload && systemctl enable smt4.service
  echo "SMT now: $(ppc64_cpu --smt) (persistent via smt4.service)"
fi

echo
echo "OS preparation done. Next:  common/scripts/check_power.sh"
