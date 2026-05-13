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
# Interleave regions so a quota-exhausted region (PREEMPTIBLE_NVIDIA_A100_GPUS
# is a per-region quota of 16; one peer A100x8 workload + ours = full) falls
# through to a region with free quota in one hop. Every zone here MUST have
# the a2-highgpu-8g shape — verified via `gcloud compute machine-types list
# --filter=name=a2-highgpu-8g` on 2026-05-13. us-east4/us-east5 do NOT have
# the shape so they are deliberately excluded.
ZONE_LIST="${ZONE_LIST:-us-east1-b us-central1-c us-west1-b europe-west4-a us-central1-a us-west4-b us-central1-b us-central1-f us-west3-b europe-west4-b}"
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
# `LORA_PACK_FILE` lets the multi-LoRA mode read a multi-line list (one
# `<name>\t<gcs-uri>` per line) without smashing it into the comma-
# separated `--metadata` argument. We use `--metadata-from-file=lora-pack`.
extra_metadata=()
metadata_from_file_args=()
if [ -n "${LORA_PACK_FILE:-}" ]; then
  [ -f "$LORA_PACK_FILE" ] || { echo "LORA_PACK_FILE=$LORA_PACK_FILE not found" >&2; exit 1; }
  metadata_from_file_args+=("--metadata-from-file=startup-script=$STARTUP_SCRIPT,lora-pack=$LORA_PACK_FILE")
  extra_metadata+=("lora-max-rank=${LORA_MAX_RANK:-16}")
else
  metadata_from_file_args+=("--metadata-from-file=startup-script=$STARTUP_SCRIPT")
  if [ -n "${LORA_GCS_URI:-}" ]; then
    extra_metadata+=("lora-gcs-uri=${LORA_GCS_URI}")
    extra_metadata+=("lora-name=${LORA_NAME:-v2}")
    extra_metadata+=("lora-max-rank=${LORA_MAX_RANK:-16}")
  fi
fi
metadata_arg=""
if [ "${#extra_metadata[@]}" -gt 0 ]; then
  metadata_arg="--metadata=$(IFS=,; echo "${extra_metadata[*]}")"
fi

last_err=""
for zone in $ZONE_LIST; do
  echo "[create-vm] trying $zone..."
  # `gcloud compute instances create` prints both progress text ("Created [...]")
  # and the --format output to the same stream. Send progress to stderr-captured
  # buf and keep only the IP on stdout by piping through tail.
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
    "${metadata_from_file_args[@]}" \
    ${metadata_arg:+"$metadata_arg"} \
    --format='value(networkInterfaces[0].accessConfigs[0].natIP)' 2>&1); then
    # Strip progress chatter and pick the line that looks like a public IPv4.
    ip=$(echo "$out" | grep -oE '([0-9]{1,3}\.){3}[0-9]{1,3}' | tail -1)
    if [ -z "$ip" ]; then
      echo "[create-vm] could not parse IP out of create output; raw:" >&2
      echo "$out" >&2
      exit 1
    fi
    echo "[create-vm] created in $zone, external IP=$ip"
    echo "ZONE=$zone"
    echo "NAME=$NAME"
    echo "EXTERNAL_IP=$ip"
    exit 0
  fi
  last_err="$out"
  # Stockout AND quota-exhausted signals both mean "try elsewhere". Stockout
  # is zone-scoped; quota is region-scoped so the next us-central1 zone won't
  # help, but a region-interleaved ZONE_LIST hops to a fresh region anyway.
  # Wording varies (STOCKOUT, RESOURCE_EXHAUSTED, resource_availability,
  # "enough resources", QUOTA_EXCEEDED, "limit exceeded", references to a
  # per-region quota metric like preemptible_nvidia_a100_gpus). Normalize
  # whitespace then case-insensitive grep.
  norm_err="$(echo "$last_err" | tr -s '\n\t ' ' ')"
  if echo "$norm_err" | grep -qiE \
      "(stockout|resource_exhausted|resource_availability|enough resources|quota_exceeded|limit exceeded|preemptible_nvidia|nvidia_a100_gpus|does not exist in zone|machine type with name)"; then
    echo "[create-vm] capacity/quota/availability issue in $zone; trying next zone"
    continue
  fi
  echo "[create-vm] non-stockout error in $zone:" >&2
  echo "$last_err" | tail -5 >&2
  exit 1
done

echo "[create-vm] no zone in {$ZONE_LIST} has capacity" >&2
echo "$last_err" | tail -5 >&2
exit 1
