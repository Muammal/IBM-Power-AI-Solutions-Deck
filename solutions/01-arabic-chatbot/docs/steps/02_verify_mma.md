# Step 02 — Prepare the OS and check MMA

## Goal

Install the system packages (Python 3.12, a modern GCC, git, cmake) and confirm that the CPU
exposes **MMA** (Matrix-Multiply Assist). MMA is the Power10/Power11 matrix engine that the
IBM-built PyTorch and llama.cpp use to make CPU inference fast.

## Files created in this step

| File | Purpose |
|------|---------|
| `scripts/02_os_prep.sh` | Installs OS packages with `dnf`; `--smt4` also switches to SMT4 |
| `scripts/check_power.sh` | Read-only health check: arch, CPU, MMA, SMT, RAM, Python. Exit code 0/1 |
| `docs/steps/02_verify_mma.md` | This document |

## Commands to run (copy-paste ready)

Copy the project to the LPAR first (from your laptop):

```bash
scp -r power-chatbot root@<LPAR_IP>:/opt/
```

Then on the LPAR:

```bash
cd /opt/power-chatbot
chmod +x scripts/*.sh
sudo ./scripts/02_os_prep.sh --smt4
./scripts/check_power.sh
```

### What is SMT4 and why?

A Power10 core can run 8 hardware threads (SMT8). The threads share the core's MMA units,
so for matrix-heavy work like LLM inference, 4 threads per core usually gives the same or
better throughput with less contention. Step 08 lets you measure it on your own LPAR.

`ppc64_cpu --smt=4` is lost at reboot. To make it permanent, create a small systemd unit:

```bash
sudo tee /etc/systemd/system/smt4.service >/dev/null <<'EOF'
[Unit]
Description=Set SMT4 for LLM inference
After=multi-user.target
[Service]
Type=oneshot
ExecStart=/usr/sbin/ppc64_cpu --smt=4
[Install]
WantedBy=multi-user.target
EOF
sudo systemctl enable --now smt4.service
```

## Expected output

```
== Architecture
  [ OK ] uname -m = ppc64le
== CPU model (lscpu)
  CPU(s):                64
  Model name:            Power11 (architected), altivec supported
  Thread(s) per core:    4
  ...
  [ OK ] CPU is Power11 (architected), altivec supported
== MMA (Matrix-Multiply Assist)
  AT_HWCAP2:       arch_3_1 htm-nosc vcrypto tar isel ebb dscr htm arch_3_00 mma
  [ OK ] MMA is available
== SMT
  SMT=4
  [ OK ] SMT4 (recommended)
== Memory
  62 GB total
  [ OK ] enough for a 3B model
== Python
  [ OK ] Python 3.12.x

RESULT: PASS
```

## How to verify

The single command that must pass before Step 03:

```bash
./scripts/check_power.sh && echo STEP02_OK
```

You can also run the raw MMA check by hand. The line must contain the word `mma`:

```bash
LD_SHOW_AUXV=1 /bin/true | grep HWCAP2
```

## Common errors and fixes

| Symptom | Cause | Fix |
|---------|-------|-----|
| `No match for argument: python3.12` | RHEL older than 9.4, or AppStream disabled | `sudo subscription-manager repos --enable rhel-9-for-ppc64le-appstream-rpms`; update to RHEL 9.4+ |
| `No match for argument: gcc-toolset-13` | Same as above | Same as above |
| HWCAP2 has no `mma` | Power9 hardware, or LPAR in **Power9 compatibility mode** | On HMC set the LPAR processor compatibility mode to Power10/Power11 (or "default") and reboot |
| `ppc64_cpu: command not found` | powerpc-utils missing | `sudo dnf install -y powerpc-utils` |
| `ppc64_cpu --smt=4` → "Machine is not SMT capable" | Running in a container/VM without access | Run it on the LPAR host, as root |
| `[WARN] under 32 GB` | Small LPAR | Use a 3B model in bfloat16, or the GGUF backend (Step 09) |
