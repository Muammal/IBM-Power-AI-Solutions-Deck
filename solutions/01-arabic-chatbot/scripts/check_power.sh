#!/usr/bin/env bash
# Checks that this machine is ready for CPU inference on IBM Power.
# Exit code 0 = all required checks passed, 1 = something required is missing.
# Usage: ./scripts/check_power.sh
set -uo pipefail

fail=0
ok()   { printf '  [ OK ] %s\n' "$1"; }
warn() { printf '  [WARN] %s\n' "$1"; }
bad()  { printf '  [FAIL] %s\n' "$1"; fail=1; }

echo "== Architecture"
arch=$(uname -m)
[[ "$arch" == "ppc64le" ]] && ok "uname -m = ppc64le" || bad "uname -m = $arch (need ppc64le)"

echo "== CPU model (lscpu)"
lscpu | grep -E '^(Model name|Thread\(s\) per core|Core\(s\) per socket|Socket\(s\)|CPU\(s\)):' | sed 's/^/  /'
model=$(lscpu | awk -F: '/^Model name/ {gsub(/^ +/,"",$2); print $2}')
if [[ "$model" =~ POWER1[0-9] || "$model" =~ Power1[0-9] ]]; then
  ok "CPU is $model"
else
  warn "CPU is '$model'. MMA needs Power10 or newer."
fi

echo "== MMA (Matrix-Multiply Assist)"
# The kernel advertises CPU features in the ELF auxiliary vector. AT_HWCAP2 must list "mma".
hwcap2=$(LD_SHOW_AUXV=1 /bin/true | grep HWCAP2)
echo "  $hwcap2"
if grep -qw mma <<<"$hwcap2"; then
  ok "MMA is available"
else
  bad "MMA not found in HWCAP2 (needs Power10+, and the LPAR must not run in Power9 compatibility mode)"
fi

echo "== SMT"
if command -v ppc64_cpu >/dev/null; then
  smt=$(ppc64_cpu --smt 2>/dev/null)
  echo "  $smt"
  [[ "$smt" == "SMT=4" ]] && ok "SMT4 (recommended)" || warn "Recommended: sudo ppc64_cpu --smt=4"
else
  warn "ppc64_cpu not found (dnf install powerpc-utils)"
fi

echo "== NUMA"
nodes=$(lscpu | awk -F: '/^NUMA node\(s\)/ {gsub(/ /,"",$2); print $2}')
echo "  ${nodes:-1} NUMA node(s)"
if [[ "${nodes:-1}" -gt 1 ]]; then
  warn "LPAR spans ${nodes} NUMA nodes: consider pinning the service to one node (numactl --cpunodebind=0 --membind=0), see Step 08"
else
  ok "single NUMA node"
fi

echo "== Memory"
mem_gb=$(awk '/MemTotal/ {printf "%d", $2/1024/1024}' /proc/meminfo)
echo "  ${mem_gb} GB total"
(( mem_gb >= 30 )) && ok "enough for a 3B model" || warn "under 32 GB: use a 3B model in bfloat16 or a GGUF file"

echo "== Python"
if command -v python3.12 >/dev/null; then ok "$(python3.12 --version)"; else bad "python3.12 missing (run scripts/02_os_prep.sh)"; fi

echo
if (( fail )); then echo "RESULT: FAIL"; exit 1; else echo "RESULT: PASS"; fi
