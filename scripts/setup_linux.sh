#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

pip install \
  vllm \
  fastapi \
  uvicorn \
  httpx \
  prometheus-client \
  pydantic-settings \
  numpy \
  pandas

docker compose up -d

echo "Setup complete."
echo "Start vLLM: ./scripts/start_vllm.sh"
echo "Start gateway: uvicorn gateway.main:app --host 0.0.0.0 --port 8000"

