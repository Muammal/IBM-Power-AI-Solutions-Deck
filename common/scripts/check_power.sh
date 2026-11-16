#!/usr/bin/env bash
# C2 — Read-only readiness check for CPU AI on IBM Power10/Power11.
# Exit code 0 = required checks passed, 1 = something required is missing.
# Usage: common/scripts/check_power.sh [--min-ram-gb N]
set -uo pipefail

MIN_RAM=${2:-30}
[[ "${1:-}" == "--min-ram-gb" ]] || MIN_RAM=30
fail=0
ok()   { printf '  [ OK ] %s\n' "$1"; }
warn() { printf '  [WARN] %s\n' "$1"; }
bad()  { printf '  [FAIL] %s\n' "$1"; fail=1; }

echo "== Architecture"
arch=$(uname -m)
[[ "$arch" == "ppc64le" ]] && ok "uname -m = ppc64le" || bad "uname -m = $arch (need ppc64le)"

echo "== CPU"
lscpu | grep -E '^(Model name|Thread\(s\) per core|Core\(s\) per socket|Socket\(s\)|CPU\(s\)):' | sed 's/^/  /'
model=$(lscpu | awk -F: '/^Model name/ {gsub(/^ +/,"",$2); print $2}')
[[ "$model" =~ [Pp][Oo][Ww][Ee][Rr]1[0-9] ]] && ok "CPU is $model" || warn "CPU is '$model'. MMA needs Power10 or newer."

echo "== MMA (Matrix-Multiply Assist)"
# The kernel lists CPU features in the ELF auxiliary vector; AT_HWCAP2 must contain "mma".
hwcap2=$(LD_SHOW_AUXV=1 /bin/true | grep HWCAP2)
echo "  $hwcap2"
grep -qw mma <<<"$hwcap2" && ok "MMA available" \
  || bad "MMA missing from HWCAP2 (needs Power10+, LPAR must not run in Power9 compatibility mode)"

echo "== SMT"
if command -v ppc64_cpu >/dev/null; then
  smt=$(ppc64_cpu --smt 2>/dev/null); echo "  $smt"
  [[ "$smt" == "SMT=4" ]] && ok "SMT4 (recommended for inference)" || warn "Recommended: sudo ppc64_cpu --smt=4"
else
  warn "ppc64_cpu not found (dnf install powerpc-utils)"
fi

echo "== Physical cores (what the AI runtimes should use as thread count)"
cores=$(( $(lscpu | awk -F: '/^Core\(s\) per socket/ {print $2+0}') * $(lscpu | awk -F: '/^Socket\(s\)/ {print $2+0}') ))
echo "  ${cores} cores"

echo "== NUMA"
nodes=$(lscpu | awk -F: '/^NUMA node\(s\)/ {gsub(/ /,"",$2); print $2}')
echo "  ${nodes:-1} NUMA node(s)"
[[ "${nodes:-1}" -gt 1 ]] && warn "Spans ${nodes} NUMA nodes: pin services with NUMA_NODE=0 (see common/README.md C5)" || ok "single NUMA node"

echo "== Memory"
mem_gb=$(awk '/MemTotal/ {printf "%d", $2/1024/1024}' /proc/meminfo)
echo "  ${mem_gb} GB"
(( mem_gb >= MIN_RAM )) && ok ">= ${MIN_RAM} GB" || warn "under ${MIN_RAM} GB: use the 3B model / Q4_0 GGUF"

echo "== Toolchain"
command -v python3.12 >/dev/null && ok "$(python3.12 --version)" || bad "python3.12 missing (run common/scripts/os_prep.sh)"
[[ -f /opt/rh/gcc-toolset-13/enable ]] && ok "gcc-toolset-13 present" || warn "gcc-toolset-13 missing (needed to build llama.cpp/whisper.cpp)"

echo
if (( fail )); then echo "RESULT: FAIL"; exit 1; else echo "RESULT: PASS"; fi
