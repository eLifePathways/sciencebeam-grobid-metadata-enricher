#!/usr/bin/env bash
# Slim startup script for VMs created from the baked image-family=qwen-bench.
# Docker, nvidia-container-toolkit, vLLM image, and /var/hf-cache are all
# already present, so this script just launches the 8 vLLM workers and
# blocks on health.
#
# Optional LoRA serving — two modes:
#   single LoRA: set instance metadata `lora-gcs-uri` (+ `lora-name`,
#                `lora-max-rank`). Adapter served under `lora-name`.
#   multi-LoRA: set instance metadata `lora-pack`, a newline-separated
#                list of `<name><TAB><gcs-uri>` entries. Each is rsynced
#                into /var/lora/<name>/ and registered as a
#                --lora-modules entry. `lora-max-rank` still applies and
#                must be >= the max rank across all adapters.
# The bench pool JSON's "deployment" (or its STEP_LORA_MAP_JSON router)
# must use those names so the OpenAI `model` field selects them.

set -euo pipefail

IMAGE="vllm/vllm-openai:latest"

meta() {
  curl -sf -H 'Metadata-Flavor: Google' \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1" || true
}

# Base model can be overridden by instance metadata `qwen-model`. The baked
# image only pre-caches Qwen2.5-7B-Instruct, so non-default models pay a
# one-time HF download (~18GB for the 9B class) at first boot.
MODEL_FROM_META="$(meta qwen-model)"
MODEL="${MODEL_FROM_META:-${QWEN_MODEL:-Qwen/Qwen2.5-7B-Instruct}}"

LORA_GCS_URI="$(meta lora-gcs-uri)"
LORA_NAME="$(meta lora-name)"; LORA_NAME="${LORA_NAME:-v2}"
LORA_MAX_RANK="$(meta lora-max-rank)"; LORA_MAX_RANK="${LORA_MAX_RANK:-16}"
LORA_PACK="$(meta lora-pack)"

lora_args=()
modules=()

if [ -n "$LORA_PACK" ]; then
  # multi-LoRA: each line is "<name>\t<gcs-uri>"
  count=0
  while IFS=$'\t' read -r name uri; do
    [ -z "$name" ] && continue
    mkdir -p "/var/lora/$name"
    echo "[startup] syncing $name from $uri"
    gsutil -m rsync -d -r "$uri" "/var/lora/$name/"
    modules+=("$name=/var/lora/$name")
    count=$((count + 1))
  done <<< "$LORA_PACK"
  lora_args=(
    --enable-lora
    --lora-modules "${modules[@]}"
    --max-lora-rank "$LORA_MAX_RANK"
    --max-loras "$count"
  )
elif [ -n "$LORA_GCS_URI" ]; then
  mkdir -p "/var/lora/${LORA_NAME}"
  gsutil -m rsync -d -r "$LORA_GCS_URI" "/var/lora/${LORA_NAME}/"
  lora_args=(
    --enable-lora
    --lora-modules "${LORA_NAME}=/var/lora/${LORA_NAME}"
    --max-lora-rank "$LORA_MAX_RANK"
  )
fi

for i in 0 1 2 3 4 5 6 7; do
  port=$((8000 + i))
  docker run -d --restart=no \
    --name "vllm-$i" \
    --gpus "\"device=$i\"" \
    --shm-size=8g \
    -v /var/hf-cache:/root/.cache/huggingface \
    -v /var/lora:/var/lora:ro \
    -p "${port}:8000" \
    "$IMAGE" \
    --model "$MODEL" \
    --served-model-name "$MODEL" \
    --enable-prefix-caching \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    "${lora_args[@]}"
done

deadline=$(( $(date +%s) + 300 ))   # 5-min budget; warm cache makes this fast
ready=0
while [ "$(date +%s)" -lt "$deadline" ] && [ "$ready" -lt 8 ]; do
  ready=0
  for i in 0 1 2 3 4 5 6 7; do
    port=$((8000 + i))
    if curl -sf "http://127.0.0.1:${port}/health" >/dev/null; then
      ready=$((ready + 1))
    fi
  done
  sleep 3
done

if [ "$ready" -lt 8 ]; then
  echo "Only $ready/8 replicas healthy after warmup window" >&2
  exit 1
fi

touch /run/qwen-ready
