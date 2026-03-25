#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"
AUTH_TOKEN="${AUTH_TOKEN:-your-secret-token-here}"
MODEL="${MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"

curl -sS "${BASE_URL}/health"
echo

curl -sS "${BASE_URL}/v1/chat/completions" \
  -H "Authorization: Bearer ${AUTH_TOKEN}" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${MODEL}\",
    \"messages\": [{\"role\":\"user\",\"content\":\"Say hello in one sentence.\"}],
    \"max_tokens\": 32
  }"
echo

