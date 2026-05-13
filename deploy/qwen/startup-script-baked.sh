#!/usr/bin/env bash
# Slim startup script for VMs created from the baked image-family=qwen-bench.
# Docker, nvidia-container-toolkit, vLLM image, and /var/hf-cache are all
# already present, so this script just launches the 8 vLLM workers and
# blocks on health.

set -euo pipefail

MODEL="${QWEN_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
IMAGE="vllm/vllm-openai:latest"

for i in 0 1 2 3 4 5 6 7; do
  port=$((8000 + i))
  docker run -d --restart=no \
    --name "vllm-$i" \
    --gpus "\"device=$i\"" \
    --shm-size=8g \
    -v /var/hf-cache:/root/.cache/huggingface \
    -p "${port}:8000" \
    "$IMAGE" \
    --model "$MODEL" \
    --served-model-name "$MODEL" \
    --enable-prefix-caching \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90
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
