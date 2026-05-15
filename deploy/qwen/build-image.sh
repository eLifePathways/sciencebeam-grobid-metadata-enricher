#!/usr/bin/env bash
# Build a custom GCE image with vLLM + Qwen3-8B pre-staged.
# After this runs, on-demand spin-up of qwen-bench-* drops from ~10 min
# (image pull + HF download + 8x vLLM cold start) to ~2 min (just the 8
# vLLM cold starts, against an already-warm shared cache on disk).
#
# Override the model with `MODEL=...` to bake a different base. Set the
# model id to match the bench's QWEN_MODEL — mismatch costs the bench an
# ~15 GB cold HF pull on first boot.
#
# Idempotent: re-running creates qwen-bench-vN+1 alongside existing images.
#
# Cost: ~$1-2 of spot-A100 burn (~15 min on a2-highgpu-1g while we install
# bits + snapshot the disk). The bake VM is auto-deleted on completion.
#
# Usage:
#   PROJECT=semiotic-garden-477923-u8 bash deploy/qwen/build-image.sh

set -euo pipefail

: "${PROJECT:?Set PROJECT=<gcp-project-id>}"
ZONE="${ZONE:-us-central1-c}"
BAKE_INSTANCE="${BAKE_INSTANCE:-qwen-image-bake}"
IMAGE_NAME="${IMAGE_NAME:-qwen-bench-$(date -u +%Y%m%d-%H%M)}"
IMAGE_FAMILY="${IMAGE_FAMILY:-qwen-bench}"
SOURCE_IMAGE_FAMILY="${SOURCE_IMAGE_FAMILY:-common-cu129-ubuntu-2204-nvidia-580}"
SOURCE_IMAGE_PROJECT="${SOURCE_IMAGE_PROJECT:-deeplearning-platform-release}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:latest}"

# We don't need 8 A100s to bake; one is enough to test the nvidia runtime.
BAKE_MACHINE="${BAKE_MACHINE:-a2-highgpu-1g}"
BAKE_ACCEL="${BAKE_ACCEL:-type=nvidia-tesla-a100,count=1}"

trap 'gcloud compute instances delete "$BAKE_INSTANCE" --zone="$ZONE" --project="$PROJECT" --quiet >/dev/null 2>&1 || true' EXIT

echo "[bake] creating builder $BAKE_INSTANCE in $ZONE"
gcloud compute instances create "$BAKE_INSTANCE" \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --machine-type="$BAKE_MACHINE" \
  --provisioning-model=SPOT \
  --instance-termination-action=DELETE \
  --max-run-duration=3600s \
  --accelerator="$BAKE_ACCEL" \
  --image-family="$SOURCE_IMAGE_FAMILY" \
  --image-project="$SOURCE_IMAGE_PROJECT" \
  --boot-disk-size=200GB \
  --boot-disk-type=pd-ssd \
  --quiet

echo "[bake] waiting for SSH to come up"
for i in $(seq 1 30); do
  if gcloud compute ssh "$BAKE_INSTANCE" --zone="$ZONE" --project="$PROJECT" \
       --command='echo ready' --quiet >/dev/null 2>&1; then break; fi
  sleep 5
done

echo "[bake] running bootstrap + pre-pull"
gcloud compute ssh "$BAKE_INSTANCE" --zone="$ZONE" --project="$PROJECT" --command="
set -euo pipefail

# Install Docker + NVIDIA Container Toolkit
if ! command -v docker >/dev/null; then
  sudo apt-get update -qq
  sudo apt-get install -yqq docker.io
fi
if ! dpkg -l nvidia-container-toolkit 2>/dev/null | grep -q ii; then
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --batch --yes --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  sudo apt-get update -qq
  sudo apt-get install -yqq nvidia-container-toolkit || true
fi
if ! grep -q '\"nvidia\"' /etc/docker/daemon.json 2>/dev/null; then
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
fi

# Pre-pull the vLLM image so the cold start skips the ~50 GB extraction
sudo docker pull '$VLLM_IMAGE'

# Pre-cache Qwen weights in /var/hf-cache; on boot the vLLM containers
# mount this read-write and skip the ~15 GB HF download.
sudo mkdir -p /var/hf-cache
sudo docker run --rm \
  -v /var/hf-cache:/root/.cache/huggingface \
  --entrypoint python3 '$VLLM_IMAGE' \
  -c 'from huggingface_hub import snapshot_download; snapshot_download(\"'$MODEL'\")'

# Smoke-test that the nvidia runtime works under docker
sudo docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi -L | head -1
" 2>&1 | tail -20

echo "[bake] stopping VM (image must be created from a stopped disk)"
gcloud compute instances stop "$BAKE_INSTANCE" --zone="$ZONE" --project="$PROJECT" --quiet

echo "[bake] creating image $IMAGE_NAME (family=$IMAGE_FAMILY)"
gcloud compute images create "$IMAGE_NAME" \
  --project="$PROJECT" \
  --source-disk="$BAKE_INSTANCE" \
  --source-disk-zone="$ZONE" \
  --family="$IMAGE_FAMILY" \
  --description="vLLM $VLLM_IMAGE + $MODEL pre-staged. Baked from $SOURCE_IMAGE_FAMILY at $(date -u +%FT%TZ)."

echo "[bake] DONE"
echo "  image:        $IMAGE_NAME"
echo "  family:       $IMAGE_FAMILY"
echo "  use with:     gcloud compute instances create ... --image-family=$IMAGE_FAMILY"
