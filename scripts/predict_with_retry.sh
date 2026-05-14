#!/usr/bin/env bash
# Run benchmarks.predict against a Qwen spot cluster with re-provision
# on preemption. The Qwen 8xA100 spot VM gets reclaimed by GCP fairly
# often during peak demand; without this wrapper a mid-run preemption
# loses all the predict work and the workflow has to be re-triggered.
#
# Inputs (via env):
#   PROJECT, NAME              — passed through to deploy/qwen/create-vm.sh
#   LORA_PACK_FILE, LORA_MAX_RANK, QWEN_MODEL
#                              — passed through on re-provision
#   POOL_PATH                  — pool JSON path (script rewrites it on re-provision)
#   BENCH_CONFIG, BENCH_MODE, BENCH_OUT
#                              — predict.py args
#   STEP_LORA_MAP_FILE         — optional; STEP_LORA_MAP_JSON env will be cat'd from here
#   LLM_CHAT_TEMPLATE_KWARGS_JSON
#                              — optional; passed through to predict.py
#   MAX_ATTEMPTS               — default 3
#   PREDICT_DEADLINE_SECONDS   — outer timeout per attempt (default 2700 = 45min)

set -uo pipefail

: "${PROJECT:?}" "${NAME:?}" "${LORA_PACK_FILE:?}" "${QWEN_MODEL:?}"
: "${POOL_PATH:?}" "${BENCH_CONFIG:?}" "${BENCH_MODE:?}" "${BENCH_OUT:?}"
LORA_MAX_RANK="${LORA_MAX_RANK:-16}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"
# 90-min cap per attempt: Qwen3.5 in --enforce-eager mode takes ~45-50 min
# on the 149-doc smoke bench (~2-3x slower than Qwen2.5 with cuda graphs).
PREDICT_DEADLINE_SECONDS="${PREDICT_DEADLINE_SECONDS:-5400}"

current_zone_file="/tmp/predict_retry_zone"

predict_once() {
  local extra_envs=()
  if [ -n "${STEP_LORA_MAP_FILE:-}" ] && [ -f "$STEP_LORA_MAP_FILE" ]; then
    extra_envs+=("STEP_LORA_MAP_JSON=$(cat "$STEP_LORA_MAP_FILE")")
  fi
  if [ -n "${LLM_CHAT_TEMPLATE_KWARGS_JSON:-}" ]; then
    extra_envs+=("LLM_CHAT_TEMPLATE_KWARGS_JSON=$LLM_CHAT_TEMPLATE_KWARGS_JSON")
  fi
  timeout "$PREDICT_DEADLINE_SECONDS" env "${extra_envs[@]}" \
    uv run python -m benchmarks.predict \
      --config "$BENCH_CONFIG" \
      --mode   "$BENCH_MODE" \
      --out    "$BENCH_OUT" \
      --pool-path "$POOL_PATH"
}

reprovision() {
  echo "[retry] re-provisioning $NAME..."
  local out
  out=$(LORA_PACK_FILE="$LORA_PACK_FILE" \
        LORA_MAX_RANK="$LORA_MAX_RANK" \
        QWEN_MODEL="$QWEN_MODEL" \
        PROJECT="$PROJECT" \
        NAME="$NAME" \
        bash deploy/qwen/create-vm.sh)
  echo "$out"
  local ip zone
  ip=$(echo "$out" | awk -F= '/^EXTERNAL_IP=/ {print $2}' | tail -1)
  zone=$(echo "$out" | awk -F= '/^ZONE=/ {print $2}' | tail -1)
  if [ -z "$ip" ] || [ -z "$zone" ]; then
    echo "[retry] create-vm did not return ip/zone" >&2
    return 1
  fi
  echo "$zone" > "$current_zone_file"
  python3 - "$ip" "$QWEN_MODEL" <<'PY' > "$POOL_PATH"
import json, sys
ip, model = sys.argv[1], sys.argv[2]
print(json.dumps([
    {"id": f"qwen-{i}",
     "endpoint": f"http://{ip}:{8000+i}",
     "deployment": model,
     "apiKey": "local",
     "apiVersion": "unused",
     "kind": "openai"}
    for i in range(8)
], indent=2))
PY
  bash scripts/wait_vllm.sh "$ip" "$QWEN_MODEL"
}

vm_is_alive() {
  local zone
  zone=$(cat "$current_zone_file" 2>/dev/null || echo "")
  [ -n "$zone" ] || return 1
  gcloud compute instances describe "$NAME" \
    --project="$PROJECT" --zone="$zone" \
    --format='value(status)' 2>/dev/null | grep -qE 'RUNNING|STAGING|PROVISIONING'
}

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  echo "=== predict attempt $attempt/$MAX_ATTEMPTS ==="
  # NB: capture exit code BEFORE any `if` test. Bash sets $? to 0 after a
  # failed `if cmd; then ...; fi` (no branch executed), masking the real
  # exit code. Previously this made a `timeout`-killed predict look like
  # rc=0 and the wrapper bailed instead of retrying.
  predict_once
  rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "[retry] predict succeeded on attempt $attempt"
    exit 0
  fi
  echo "[retry] predict failed rc=$rc"
  if [ "$attempt" -lt "$MAX_ATTEMPTS" ]; then
    if ! vm_is_alive; then
      echo "[retry] VM gone — re-provisioning"
      if ! reprovision; then
        echo "[retry] re-provision failed; aborting" >&2
        exit 1
      fi
    else
      echo "[retry] VM still alive but predict failed; bailing (likely a code error, not preemption)" >&2
      exit "$rc"
    fi
  fi
done
echo "[retry] exhausted $MAX_ATTEMPTS attempts" >&2
exit 1
