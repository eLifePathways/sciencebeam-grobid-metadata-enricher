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
# SPOT (default) cycles through zones on stockout; ON_DEMAND skips the spot
# preemption + cancellation modes for one guaranteed-available run (~$15/hr
# incremental for a2-highgpu-8g).
PROVISIONING_MODEL="${PROVISIONING_MODEL:-SPOT}"
# Hard cap per zone attempt; gcloud waits synchronously on operation
# completion (STAGING -> RUNNING or failure) which can hang for minutes
# when GCP is churning. 180s lets a legitimate provision land while
# bounding the bad case so we cycle through ZONE_LIST in worst-case
# 180s * len(ZONE_LIST) instead of unbounded.
PER_ZONE_TIMEOUT="${PER_ZONE_TIMEOUT:-180}"

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
if [ -n "${QWEN_MODEL:-}" ]; then
  extra_metadata+=("qwen-model=${QWEN_MODEL}")
fi
if [ -n "${VLLM_CACHE_GCS:-}" ]; then
  extra_metadata+=("vllm-cache-gcs=${VLLM_CACHE_GCS}")
fi
metadata_arg=""
if [ "${#extra_metadata[@]}" -gt 0 ]; then
  metadata_arg="--metadata=$(IFS=,; echo "${extra_metadata[*]}")"
fi

provisioning_args=()
case "$PROVISIONING_MODEL" in
  SPOT)
    provisioning_args+=(--provisioning-model=SPOT
                        --instance-termination-action=DELETE
                        --max-run-duration="${MAX_RUN_SECONDS}s")
    ;;
  ON_DEMAND)
    # No spot flags; gcloud defaults to STANDARD on-demand provisioning.
    # max-run-duration is incompatible with STANDARD, so drop it too.
    echo "[create-vm] using ON_DEMAND provisioning (no spot, no preemption)"
    ;;
  *)
    echo "[create-vm] unknown PROVISIONING_MODEL=$PROVISIONING_MODEL" >&2; exit 2
    ;;
esac
# A100 (and every other accelerator-attached VM) cannot live-migrate, so
# maintenance-policy MUST be TERMINATE. SPOT's --instance-termination-action
# implicitly forces this; ON_DEMAND defaults to MIGRATE, which gcloud rejects
# for any GPU-bearing instance ("onHostMaintenance must be TERMINATE").
provisioning_args+=(--maintenance-policy=TERMINATE)

last_err=""
for zone in $ZONE_LIST; do
  echo "[create-vm] trying $zone (mode=$PROVISIONING_MODEL, timeout=${PER_ZONE_TIMEOUT}s)..."
  # `gcloud compute instances create` prints both progress text ("Created [...]")
  # and the --format output to the same stream. Wrap in `timeout` so a hung
  # create operation doesn't burn the whole multi-zone budget. NB: capture
  # rc explicitly (not via `if cmd; then`) because $? after a failed
  # `if cmd; then ...; fi` is the if-statement's exit code (always 0).
  out=$(timeout "${PER_ZONE_TIMEOUT}s" gcloud compute instances create "$NAME" \
    --project="$PROJECT" \
    --zone="$zone" \
    --machine-type="$MACHINE" \
    "${provisioning_args[@]}" \
    --accelerator="$ACCEL" \
    --image-family="$IMAGE_FAMILY" \
    --image-project="$IMAGE_PROJECT" \
    --boot-disk-size=200GB \
    --boot-disk-type=pd-ssd \
    --tags=qwen-bench \
    "${metadata_from_file_args[@]}" \
    ${metadata_arg:+"$metadata_arg"} \
    --format='value(networkInterfaces[0].accessConfigs[0].natIP)' 2>&1) && rc=0 || rc=$?
  if [ "$rc" -eq 0 ]; then
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
  # rc=124 = our `timeout` killed gcloud (per-zone hard cap exceeded).
  # Treat that exactly like a stockout. Best-effort fire-and-forget delete
  # of any half-created VM in this zone so a re-attempt doesn't 409 on
  # name conflict (the teardown step is also resilient to wrong zones).
  if [ "$rc" -eq 124 ]; then
    echo "[create-vm] $zone timed out after ${PER_ZONE_TIMEOUT}s; trying next zone"
    gcloud compute instances delete "$NAME" --project="$PROJECT" --zone="$zone" \
      --quiet >/dev/null 2>&1 &
    continue
  fi
  # Stockout AND quota-exhausted signals both mean "try elsewhere". Stockout
  # is zone-scoped; quota is region-scoped so the next us-central1 zone won't
  # help, but a region-interleaved ZONE_LIST hops to a fresh region anyway.
  # Wording varies (STOCKOUT, RESOURCE_EXHAUSTED, resource_availability,
  # "enough resources", QUOTA_EXCEEDED, "limit exceeded", references to a
  # per-region quota metric like preemptible_nvidia_a100_gpus). Normalize
  # whitespace then case-insensitive grep.
  # "operation was canceled" appears when GCP preempts a spot create mid-
  # staging — same intent, try a different zone instead of bailing out.
  norm_err="$(echo "$last_err" | tr -s '\n\t ' ' ')"
  if echo "$norm_err" | grep -qiE \
      "(stockout|resource_exhausted|resource_availability|enough resources|quota_exceeded|limit exceeded|preemptible_nvidia|nvidia_a100_gpus|does not exist in zone|machine type with name|operation was canceled)"; then
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
