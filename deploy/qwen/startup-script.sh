#!/usr/bin/env bash
# VM startup script for the on-demand Qwen-7B serving node (UNBAKED path).
# Target image family: deeplearning-platform-release/common-cu129-ubuntu-2204-nvidia-580
# Target shape: a2-highgpu-8g (8 x A100 40GB, spot)
#
# Use this only if the baked image-family=qwen-bench isn't available yet
# (see build-image.sh). With the baked image, prefer startup-script-baked.sh
# which skips install + pull + download (~8 min faster cold start).
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

# Install Docker + NVIDIA Container Toolkit. The DL CUDA image ships drivers
# but not docker; idempotent so re-runs / baked images are no-ops.
if ! command -v docker >/dev/null; then
  apt-get update -qq
  apt-get install -yqq docker.io
fi
if ! dpkg -l nvidia-container-toolkit 2>/dev/null | grep -q ii; then
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | gpg --batch --yes --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    > /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update -qq
  apt-get install -yqq nvidia-container-toolkit || true
fi
if ! grep -q '"nvidia"' /etc/docker/daemon.json 2>/dev/null; then
  nvidia-ctk runtime configure --runtime=docker
  systemctl restart docker
fi

mkdir -p "$HF_CACHE"
docker pull "$IMAGE"

# Pre-download once into the shared cache so the 8 vLLM containers don't
# race on the HF download. vLLM image lacks a 'python' symlink, only python3.
docker run --rm \
  -v "$HF_CACHE:/root/.cache/huggingface" \
  -e HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
  --entrypoint python3 "$IMAGE" \
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
    --gpu-memory-utilization 0.90
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
