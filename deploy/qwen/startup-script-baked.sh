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

# vLLM caches its torch.compile artifacts (~150s of work for Qwen3.5-9B:
# Dynamo bytecode transform + AOT compile + first warmup) inside the
# container at /root/.cache/vllm/torch_compile_cache. Empty on every
# fresh VM, so it recompiles from scratch.
#
# Mount /var/vllm-compile-cache from the host -> the cache path in the
# container. If `vllm-cache-gcs` metadata is set, gsutil rsync it in
# before docker so a populated cache from a previous bake VM cuts cold
# start by ~2.5 minutes.
mkdir -p /var/vllm-compile-cache
VLLM_CACHE_GCS="$(meta vllm-cache-gcs)"
if [ -n "$VLLM_CACHE_GCS" ]; then
  echo "[startup] rsyncing vLLM compile cache from $VLLM_CACHE_GCS"
  gsutil -m rsync -r "$VLLM_CACHE_GCS" /var/vllm-compile-cache/ || \
    echo "[startup] cache rsync failed; vLLM will recompile from scratch"
fi

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
  # vLLM only needs root adapter_config.json + adapter_model.safetensors +
  # tokenizer; training checkpoint-<step>/ dirs add ~600 MiB per adapter
  # and aren't read at serve time, so we exclude them via -x regex.
  count=0
  while IFS=$'\t' read -r name uri; do
    [ -z "$name" ] && continue
    mkdir -p "/var/lora/$name"
    echo "[startup] syncing $name from $uri"
    gsutil -m rsync -d -r -x '^checkpoint-[0-9]+/' "$uri" "/var/lora/$name/"
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

# Qwen3-Next family (Qwen3.x and Qwen3-Next-* — qwen3_next arch) hits
# cudaErrorNotPermitted during cuda-graph capture in vLLM 0.20.x. The plain
# qwen3 arch (Qwen/Qwen3-8B, etc.) is NOT affected and serves fine with
# full cuda graphs.
#
# vllm-graph-mode metadata (default 'auto'):
#   auto      — qwen3_next models: --enforce-eager; everything else: stock
#   no-lora-graph — qwen3_next: cudagraph_specialize_lora=false (faster
#                   but unreliable); for experimentation
#   eager     — force --enforce-eager
#   stock     — no overrides
is_qwen3_next() {
  [[ "$1" == *"Qwen3."* ]] || [[ "$1" == *"Qwen3-Next"* ]]
}
GRAPH_MODE="$(meta vllm-graph-mode)"; GRAPH_MODE="${GRAPH_MODE:-auto}"
extra_vllm_args=()
case "$GRAPH_MODE" in
  eager)
    extra_vllm_args+=(--enforce-eager)
    ;;
  stock)
    ;;
  no-lora-graph)
    if is_qwen3_next "$MODEL"; then
      extra_vllm_args+=(--compilation-config '{"cudagraph_specialize_lora": false}')
    fi
    ;;
  auto)
    is_qwen3_next "$MODEL" && extra_vllm_args+=(--enforce-eager)
    ;;
  *)
    echo "[startup] unknown vllm-graph-mode=$GRAPH_MODE; ignoring" >&2
    ;;
esac

for i in 0 1 2 3 4 5 6 7; do
  port=$((8000 + i))
  docker run -d --restart=no \
    --name "vllm-$i" \
    --gpus "\"device=$i\"" \
    --shm-size=8g \
    -v /var/hf-cache:/root/.cache/huggingface \
    -v /var/lora:/var/lora:ro \
    -v /var/vllm-compile-cache:/root/.cache/vllm/torch_compile_cache \
    -p "${port}:8000" \
    "$IMAGE" \
    --model "$MODEL" \
    --served-model-name "$MODEL" \
    --enable-prefix-caching \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    "${extra_vllm_args[@]}" \
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
