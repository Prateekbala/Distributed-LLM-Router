#!/usr/bin/env bash
# Start three vLLM OpenAI API servers on ports 8001–8003 (requires sufficient GPU memory).
set -euo pipefail

MODEL="${MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
HOST="${HOST:-0.0.0.0}"

for PORT in 8001 8002 8003; do
  echo "Starting vLLM on port ${PORT}..."
  python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --dtype auto \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 16 &
done

echo "Cluster PIDs: $!"
echo "Nodes: http://localhost:8001 http://localhost:8002 http://localhost:8003"
wait
