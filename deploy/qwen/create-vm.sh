#!/usr/bin/env bash
# Create a Qwen-bench spot VM with multi-zone stockout retry.
# Walks ZONE_LIST in order; falls through on RESOURCE_EXHAUSTED.
# Prints the VM's external IP on success.
#
# Assumes a custom image baked by build-image.sh (image-family=qwen-bench).
# Falls back to the raw DL image with a slim startup script if the baked
# family doesn't exist yet (set RAW_IMAGE_FAMILY env to force this path).

set -euo pipefail

: "${PROJECT:?Set PROJECT=<gcp-project-id>}"
NAME="${NAME:-qwen-bench-$(date -u +%Y%m%d-%H%M)}"
ZONE_LIST="${ZONE_LIST:-us-central1-a us-central1-b us-central1-c us-central1-f us-east4-c europe-west4-a}"
MACHINE="${MACHINE:-a2-highgpu-8g}"
ACCEL="${ACCEL:-type=nvidia-tesla-a100,count=8}"
MAX_RUN_SECONDS="${MAX_RUN_SECONDS:-7200}"

IMAGE_FAMILY="${IMAGE_FAMILY:-qwen-bench}"
IMAGE_PROJECT="${IMAGE_PROJECT:-$PROJECT}"
RAW_IMAGE_FAMILY="${RAW_IMAGE_FAMILY:-}"   # set to "common-cu129-ubuntu-2204-nvidia-580" to skip the baked image

if [ -n "$RAW_IMAGE_FAMILY" ]; then
  IMAGE_FAMILY="$RAW_IMAGE_FAMILY"
  IMAGE_PROJECT="deeplearning-platform-release"
  STARTUP_SCRIPT="$(dirname "$0")/startup-script.sh"
else
  STARTUP_SCRIPT="$(dirname "$0")/startup-script-baked.sh"
fi

[ -f "$STARTUP_SCRIPT" ] || { echo "startup script $STARTUP_SCRIPT not found" >&2; exit 1; }

# Optional LoRA serving: pass through to the VM as instance metadata; the
# baked startup script picks these up and configures vLLM accordingly.
extra_metadata=()
if [ -n "${LORA_GCS_URI:-}" ]; then
  extra_metadata+=("lora-gcs-uri=${LORA_GCS_URI}")
  extra_metadata+=("lora-name=${LORA_NAME:-v2}")
  extra_metadata+=("lora-max-rank=${LORA_MAX_RANK:-16}")
fi
metadata_arg=""
if [ "${#extra_metadata[@]}" -gt 0 ]; then
  metadata_arg="--metadata=$(IFS=,; echo "${extra_metadata[*]}")"
fi

last_err=""
for zone in $ZONE_LIST; do
  echo "[create-vm] trying $zone..."
  if out=$(gcloud compute instances create "$NAME" \
    --project="$PROJECT" \
    --zone="$zone" \
    --machine-type="$MACHINE" \
    --provisioning-model=SPOT \
    --instance-termination-action=DELETE \
    --max-run-duration="${MAX_RUN_SECONDS}s" \
    --accelerator="$ACCEL" \
    --image-family="$IMAGE_FAMILY" \
    --image-project="$IMAGE_PROJECT" \
    --boot-disk-size=200GB \
    --boot-disk-type=pd-ssd \
    --tags=qwen-bench \
    --metadata-from-file="startup-script=$STARTUP_SCRIPT" \
    ${metadata_arg:+"$metadata_arg"} \
    --format='value(networkInterfaces[0].accessConfigs[0].natIP)' 2>&1); then
    echo "[create-vm] created in $zone, external IP=$out"
    echo "ZONE=$zone"
    echo "NAME=$NAME"
    echo "EXTERNAL_IP=$out"
    exit 0
  fi
  last_err="$out"
  if echo "$last_err" | grep -qE "(stockout|does not have enough resources|RESOURCE_EXHAUSTED)"; then
    echo "[create-vm] stockout in $zone; trying next zone"
    continue
  fi
  echo "[create-vm] non-stockout error in $zone:" >&2
  echo "$last_err" | tail -5 >&2
  exit 1
done

echo "[create-vm] no zone in {$ZONE_LIST} has capacity" >&2
echo "$last_err" | tail -5 >&2
exit 1
