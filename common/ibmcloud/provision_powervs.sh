#!/usr/bin/env bash
# C1 — Create a RHEL 9 LPAR on IBM Power Virtual Server (PowerVS) for one of the solutions.
# Run on your laptop (Linux/macOS/WSL/Git Bash) with the IBM Cloud CLI installed.
#
# Usage:  ./provision_powervs.sh <lpar-name> <size>      size = small | medium | large
#           small  : 4 cores / 32 GB   (3B models, Solutions 05/06/07)
#           medium : 8 cores / 64 GB   (8B models, Solutions 01/02/03/04/08/09)
#           large  : 16 cores / 128 GB (30B model or several services on one LPAR)
# Required env: WORKSPACE_CRN  IMAGE_ID  SUBNET_ID  SSH_KEY_NAME
# Optional env: SYS_TYPE (default s1122 = Power11; use s1022 for Power10)  PROC_TYPE (dedicated|shared)
#
# Flag names change between power-iaas plugin versions. If one is rejected:
#   ibmcloud pi instance create --help
set -euo pipefail
NAME=${1:?lpar name}
SIZE=${2:-medium}
case "$SIZE" in
  small)  CORES=4;  MEM=32  ;;
  medium) CORES=8;  MEM=64  ;;
  large)  CORES=16; MEM=128 ;;
  *) echo "size must be small|medium|large"; exit 1 ;;
esac
: "${WORKSPACE_CRN:?export WORKSPACE_CRN (ibmcloud pi workspace list)}"
: "${IMAGE_ID:?export IMAGE_ID (RHEL 9 image in the workspace: ibmcloud pi image list)}"
: "${SUBNET_ID:?export SUBNET_ID (ibmcloud pi subnet list)}"
: "${SSH_KEY_NAME:?export SSH_KEY_NAME (ibmcloud pi ssh-key list)}"

ibmcloud plugin install power-iaas -f >/dev/null
ibmcloud pi workspace target "$WORKSPACE_CRN"

ibmcloud pi instance create "$NAME" \
  --image "$IMAGE_ID" \
  --subnets "$SUBNET_ID" \
  --key-name "$SSH_KEY_NAME" \
  --sys-type "${SYS_TYPE:-s1122}" \
  --processor-type "${PROC_TYPE:-dedicated}" \
  --processors "$CORES" \
  --memory "$MEM" \
  --storage-tier tier1

echo "Waiting for ACTIVE (can take 10-20 minutes)..."
for _ in $(seq 1 60); do
  status=$(ibmcloud pi instance get "$NAME" --json | jq -r '.status')
  echo "  status: $status"
  [[ "$status" == "ACTIVE" ]] && break
  sleep 30
done
ibmcloud pi instance get "$NAME"
echo
echo "Next: ssh to the LPAR, clone this repo to /opt/ibm-power-ai-solutions and run common/scripts/os_prep.sh"
