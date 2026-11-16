# Step 01 — Provision the Power LPAR

## Goal

Get one RHEL 9 LPAR (logical partition) on IBM Power10 or Power11 that is large enough
to run a 3B–8B chat model on CPU. You can create it on **IBM Power Virtual Server
(PowerVS)** in IBM Cloud, or ask your Power admin for an **on-premises** LPAR.

At the end of this step you can `ssh` into a RHEL 9 machine whose `uname -m` prints `ppc64le`.

## Files created in this step

| File | Purpose |
|------|---------|
| `docs/steps/01_provision_power.md` | This document (no code yet) |

## Sizing guide

The model weights must fit in RAM **plus** room for the KV cache, Python and the OS.
Rule of thumb for weight size: `parameters × bytes per parameter`
(float32 = 4 bytes, bfloat16 = 2 bytes, GGUF Q8_0 ≈ 1.06 bytes, GGUF Q4_0 ≈ 0.56 bytes).

| Model size | Example preset | Weights bf16 / fp32 | Suggested cores (dedicated) | Suggested RAM | Disk |
|-----------:|----------------|--------------------:|----------------------------:|--------------:|-----:|
| 3B | Llama-3.2-3B, Qwen2.5-3B | ~6.5 GB / ~13 GB | 4 | 32 GB | 100 GB |
| 7B | Qwen2.5-7B, ALLaM-7B | ~15 GB / ~30 GB | 8 | 64 GB | 150 GB |
| 8B | e.g. Llama-3.1-8B | ~16 GB / ~32 GB | 8 | 64 GB | 150 GB |

Notes:
* A Power10/Power11 core runs up to 8 hardware threads (SMT8). Step 02 recommends SMT4.
* **Dedicated** cores give stable benchmark numbers; shared/capped cores are cheaper but noisy.
* The Hugging Face cache (`~/.cache/huggingface`) keeps every model you try. Plan disk for 2–3 models.
* If you plan to benchmark float32 for 7B/8B (Step 08), use the 64 GB size.

## Commands to run (copy-paste ready)

### Option A — IBM Power Virtual Server with the `ibmcloud` CLI

Run these on your laptop (any OS) after installing the
[IBM Cloud CLI](https://cloud.ibm.com/docs/cli).

```bash
# 1. Log in and install the Power Virtual Server plugin
ibmcloud login --sso
ibmcloud plugin install power-iaas -f

# 2. Pick your PowerVS workspace (create one in the console first if you have none)
ibmcloud pi workspace list
ibmcloud pi workspace target <WORKSPACE_CRN>

# 3. Find the inputs you need
ibmcloud pi image list-catalog | grep -i rhel     # stock RHEL 9 images
ibmcloud pi image create <RHEL9_CATALOG_IMAGE_ID> # copy a stock image into the workspace
ibmcloud pi subnet list                           # network (subnet) ID
ibmcloud pi ssh-key list                          # or: ibmcloud pi ssh-key create mykey --key "$(cat ~/.ssh/id_rsa.pub)"

# 4. Create the LPAR (7B size shown; for 3B use --processors 4 --memory 32)
ibmcloud pi instance create chatbot-lpar \
  --image <RHEL9_IMAGE_ID> \
  --subnets <SUBNET_ID> \
  --key-name <SSH_KEY_NAME> \
  --sys-type s1122 \
  --processor-type dedicated \
  --processors 8 \
  --memory 64 \
  --storage-tier tier1

# 5. Wait until status is ACTIVE, then read the IP address
ibmcloud pi instance get chatbot-lpar
```

* `s1122` is the Power11 system type. If your region does not offer it yet, use `s1022` (Power10).
  `ibmcloud pi instance create --help` lists the system types available to you.
* Flag names change between plugin versions (`--subnets` used to be `--network`).
  If a flag is rejected, check `ibmcloud pi instance create --help`.
* Stock RHEL images on PowerVS include the RHEL subscription. Register your own image if you bring a licence.

### Option B — On-premises Power10/Power11 server

Ask your Power admin (HMC or PowerVM NovaLink) for an LPAR with:

* RHEL 9.4 or newer, ppc64le, registered with `subscription-manager` (AppStream + BaseOS repos enabled)
* Dedicated processors and memory from the sizing table above
* SSH access for your user with `sudo`

## Expected output

`ibmcloud pi instance get chatbot-lpar` shows:

```
Status            ACTIVE
Processors        8
Memory            64
Processor Type    dedicated
System Type       s1122
Networks          ... External IP: 203.0.113.10
```

## How to verify

Log in and confirm architecture, OS and size:

```bash
ssh root@<LPAR_IP>
uname -m && grep PRETTY_NAME /etc/os-release && nproc && free -g | head -2
```

Expected (numbers depend on your size; `nproc` = cores × SMT threads):

```
ppc64le
PRETTY_NAME="Red Hat Enterprise Linux 9.6 (Plow)"
64
              total        used        free      shared  buff/cache   available
Mem:             62           1          60           0           0          61
```

**Verification passes when `uname -m` prints `ppc64le` and the OS is RHEL 9.** Continue to Step 02.

## Common errors and fixes

| Symptom | Cause | Fix |
|---------|-------|-----|
| `No workspace targeted` | Step 2 skipped | `ibmcloud pi workspace target <CRN>` |
| `system type s1122 not available` | Power11 not in this region/zone | Use `--sys-type s1022` or choose another zone |
| `insufficient capacity` for dedicated cores | Pool exhausted | Try fewer cores, another zone, or `--processor-type shared` for testing |
| Cannot SSH: timeout | No public IP / wrong subnet | Attach a public subnet or use the VPN/Transit Gateway your org provides |
| `dnf` says "This system is not registered" | BYOL image | `subscription-manager register` and attach the subscription |
