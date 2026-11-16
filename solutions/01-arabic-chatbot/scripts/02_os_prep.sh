#!/usr/bin/env bash
# Step 02 — Install the OS packages the chatbot needs on RHEL 9 (ppc64le).
# Package list follows the ppc64le/pyeco Python3.12 examples (v2026.03.31).
#
# Usage:  sudo ./scripts/02_os_prep.sh            # install packages only
#         sudo ./scripts/02_os_prep.sh --smt4     # also switch the LPAR to SMT4 now
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "Please run as root (sudo $0)"; exit 1
fi

# Same distro check the pyeco install scripts use, reduced to the RHEL family.
. /etc/os-release
case "$ID" in
  rhel|centos|rocky|almalinux|fedora) ;;
  *) echo "Unsupported distribution: $ID (this template targets RHEL 9)"; exit 1 ;;
esac

if [[ "$(uname -m)" != "ppc64le" ]]; then
  echo "WARNING: architecture is $(uname -m), not ppc64le. Continuing anyway."
fi

# python3.12          -> the interpreter IBM's wheels are built for (cp312)
# python3.12-devel    -> headers, needed if pip ever has to compile something
# gcc-toolset-13      -> modern GCC with full Power10 (-mcpu=power10, MMA) support
# cmake, gcc-c++      -> needed to build llama-cpp-python in Step 09
# powerpc-utils       -> provides ppc64_cpu (SMT control)
# numactl             -> lets you pin the server to cores/memory if you want
dnf install -y \
  python3.12 python3.12-devel python3.12-pip \
  gcc gcc-c++ gcc-toolset-13 cmake make git \
  powerpc-utils numactl curl tar

if [[ "${1:-}" == "--smt4" ]]; then
  # SMT4 = 4 hardware threads per core. For LLM inference this usually beats SMT8,
  # because the matrix units (MMA) are shared per core. Not persistent across reboot,
  # see docs/steps/02_verify_mma.md for making it permanent.
  ppc64_cpu --smt=4
  echo "SMT set to: $(ppc64_cpu --smt)"
fi

echo
echo "OS preparation done. Now run:  ./scripts/check_power.sh"
