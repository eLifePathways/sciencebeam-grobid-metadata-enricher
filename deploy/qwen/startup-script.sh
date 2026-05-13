#!/usr/bin/env bash
# VM startup script for the on-demand Qwen-7B serving node.
# Target image family: deeplearning-platform-release/common-cu129-ubuntu-2204-nvidia-580
# Target shape: a2-highgpu-8g (8 x A100 40GB, spot)
#
# Brings up 8 vLLM workers, one per GPU, on ports 8000..8007 sharing a
# single host-mounted HF cache so the model weights are pulled once.
# Writes /run/qwen-ready when all 8 are healthy so the caller can poll.

set -euo pipefail

MODEL="${QWEN_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
HF_TOKEN="$(curl -sf -H 'Metadata-Flavor: Google' \
  http://metadata.google.internal/computeMetadata/v1/instance/attributes/hf-token || true)"

IMAGE="vllm/vllm-openai:latest"
HF_CACHE=/var/hf-cache

mkdir -p "$HF_CACHE"
docker pull "$IMAGE"

# Pre-download once into the shared cache so the 8 vLLM containers don't
# race on the HF download. Qwen-7B Instruct is ungated as of 2026-05; the
# HF token is optional but improves rate limits if supplied.
docker run --rm \
  -v "$HF_CACHE:/root/.cache/huggingface" \
  -e HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
  --entrypoint python "$IMAGE" \
  -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL')"

for i in 0 1 2 3 4 5 6 7; do
  port=$((8000 + i))
  docker run -d --restart=no \
    --name "vllm-$i" \
    --gpus "\"device=$i\"" \
    --shm-size=8g \
    -v "$HF_CACHE:/root/.cache/huggingface" \
    -e HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
    -p "${port}:8000" \
    "$IMAGE" \
    --model "$MODEL" \
    --served-model-name "$MODEL" \
    --enable-prefix-caching \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --disable-log-requests
done

# Block until all 8 replicas answer /health 200. With weights already on
# disk the per-replica warmup is ~60-120s; budget 15 min for slow disk.
deadline=$(( $(date +%s) + 900 ))
ready=0
while [ "$(date +%s)" -lt "$deadline" ] && [ "$ready" -lt 8 ]; do
  ready=0
  for i in 0 1 2 3 4 5 6 7; do
    port=$((8000 + i))
    if curl -sf "http://127.0.0.1:${port}/health" >/dev/null; then
      ready=$((ready + 1))
    fi
  done
  sleep 5
done

if [ "$ready" -lt 8 ]; then
  echo "Only $ready/8 replicas healthy after warmup window" >&2
  exit 1
fi

touch /run/qwen-ready
