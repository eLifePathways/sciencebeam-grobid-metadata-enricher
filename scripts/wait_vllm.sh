#!/usr/bin/env bash
# Block until all 8 vLLM workers on a Qwen spot VM respond to real
# chat-completion requests, or the deadline expires.
#
# Usage:
#   bash scripts/wait_vllm.sh <ip> <model-name> [deadline-seconds]
#
# Polling a real /v1/chat/completions catches the dead-engine case where
# /health says 200 but the engine subprocess hasn't registered the API
# routes — see deploy/qwen/startup-script-baked.sh for the symptom.

set -uo pipefail

IP="${1:?ip required}"
MODEL="${2:?model name required}"
DEADLINE_SECONDS="${3:-1500}"   # 25 min default — covers Qwen3.5 cold HF pull

payload=$(printf '{"model":"%s","messages":[{"role":"user","content":"hi"}],"max_tokens":1,"temperature":0}' "$MODEL")
deadline=$(( $(date +%s) + DEADLINE_SECONDS ))
while [ "$(date +%s)" -lt "$deadline" ]; do
  ready=0
  for i in 0 1 2 3 4 5 6 7; do
    code=$(curl -s -o /dev/null -w '%{http_code}' -m 5 \
      -X POST "http://${IP}:$((8000+i))/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      --data "$payload" || echo "000")
    [ "$code" = "200" ] && ready=$((ready+1))
  done
  echo "[$(date +%H:%M:%S)] $ready/8 vLLM ready"
  [ "$ready" -eq 8 ] && exit 0
  sleep 15
done
echo "::error::only $ready/8 vLLM replicas ready after ${DEADLINE_SECONDS}s" >&2
exit 1
