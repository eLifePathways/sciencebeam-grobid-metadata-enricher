#!/usr/bin/env bash
# Slim startup script for VMs created from the baked image-family=qwen-bench.
# Docker, nvidia-container-toolkit, vLLM image, and /var/hf-cache are all
# already present, so this script just launches the 8 vLLM workers and
# blocks on health.
#
# Optional LoRA serving: if the instance metadata `lora-gcs-uri` is set
# (e.g. gs://my-bucket/lora/v2/), the script fetches it into /var/lora/<name>
# and passes --enable-lora --lora-modules to vLLM. The adapter is served
# under the name read from metadata `lora-name` (default: "v2"). The
# bench pool JSON should use that name as its "deployment" field so the
# OpenAI-compat model= field routes through the adapter.

set -euo pipefail

MODEL="${QWEN_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
IMAGE="vllm/vllm-openai:latest"

LORA_GCS_URI="$(curl -sf -H 'Metadata-Flavor: Google' \
  http://metadata.google.internal/computeMetadata/v1/instance/attributes/lora-gcs-uri || true)"
LORA_NAME="$(curl -sf -H 'Metadata-Flavor: Google' \
  http://metadata.google.internal/computeMetadata/v1/instance/attributes/lora-name || echo v2)"
LORA_MAX_RANK="$(curl -sf -H 'Metadata-Flavor: Google' \
  http://metadata.google.internal/computeMetadata/v1/instance/attributes/lora-max-rank || echo 16)"

lora_args=()
if [ -n "$LORA_GCS_URI" ]; then
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
