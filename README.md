# Distributed LLM Router

OpenAI-compatible inference gateway that turns a single-node setup into a **fault-tolerant, observable, multi-node LLM serving system**.

## Project Goals

- Routes traffic across multiple vLLM nodes in real time.
- Supports three routing strategies: `round_robin`, `least_loaded`, `latency_based`.
- Handles failures with retries + failover (no single-node bottleneck).
- Enforces backpressure (`semaphore + queue`) to stay stable under load.
- Exposes production-ready Prometheus metrics and Grafana dashboard.
- Benchmarks include node distribution visibility (`X-Routed-Node`).

## What it is

`Client -> Router Gateway -> vLLM Node Pool`

Example pool:

- `http://localhost:8001`
- `http://localhost:8002`
- `http://localhost:8003`

The gateway keeps API compatibility with `POST /v1/chat/completions`, so existing clients continue to work.

## Quick start (5 steps)

### 1) Setup environment

```bash
./scripts/setup_linux.sh
```

### 2) Configure router

Edit `gateway/.env`:

```env
VLLM_NODES=http://localhost:8001,http://localhost:8002,http://localhost:8003
ROUTING_STRATEGY=least_loaded
NODE_TIMEOUT_SECONDS=120
MAX_RETRIES=2
HEALTH_CHECK_INTERVAL_SECONDS=15
HEALTH_FAILURE_THRESHOLD=3
GATEWAY_PORT=8000
AUTH_TOKEN=your-secret-token-here
MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.3
CONNECT_TIMEOUT_SECONDS=10
MAX_CONCURRENT_REQUESTS=16
MAX_QUEUE_SIZE=32
```

### 3) Start vLLM cluster

```bash
./scripts/start_cluster.sh
```

### 4) Start gateway

```bash
source .venv/bin/activate
uvicorn gateway.main:app --host 0.0.0.0 --port 8000
```

### 5) Smoke test

```bash
./scripts/smoke_test.sh
```

## Core endpoints

- `POST /v1/chat/completions` - OpenAI-compatible routed inference
- `GET /health` - gateway status + router config + node list
- `GET /stats` - per-node load/latency/errors/health snapshot
- `GET /metrics` - Prometheus metrics

## Benchmark

**Reference (Colab T4, single node, Qwen2.5-3B-Instruct):** ~7× higher aggregate throughput concurrent vs naive; sweep shows throughput leveling off once client concurrency exceeds server limits. [Colab reproduction](docs/COLAB_SETUP.md).


Run from the repository root with `PYTHONPATH=.`. Set **`--model`** to the same model ID as vLLM and `MODEL_NAME` in `gateway/.env`.

| Harness | Purpose |
|--------|---------|
| `benchmark.run_benchmark` | Compares sequential (**naive**) vs overlapping (**concurrent**) requests; reports throughput and latency. |
| `benchmark.run_load_sweep` | Varies client concurrency to observe scaling and saturation under gateway / vLLM limits. |

**Throughput vs sequential** (`--mode both`):

```bash
PYTHONPATH=. python -m benchmark.run_benchmark \
  --mode both \
  --num-requests 60 \
  --concurrency 12 \
  --base-url http://localhost:8000 \
  --auth-token your-secret-token-here \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --output benchmark/results/run_both.json
```

Add **`--log-node`** to summarize routing across nodes via `X-Routed-Node`.

**Concurrency sweep** (writes JSON + CSV under `benchmark/results/`):

```bash
PYTHONPATH=. python -m benchmark.run_load_sweep \
  --num-requests 60 \
  --base-url http://localhost:8000 \
  --auth-token your-secret-token-here \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --output-json benchmark/results/load_sweep.json \
  --output-csv benchmark/results/load_sweep.csv
```

Optional: **`--concurrencies 1,2,4,8`** to align sweep steps with `MAX_CONCURRENT_REQUESTS` and vLLM `--max-num-seqs`.

## Observability highlights

Prometheus + Grafana track both gateway and node behavior:

- traffic and latency: `requests_total`, `request_latency_seconds`
- overload safety: `queue_depth`, `requests_rejected_total`
- node-level routing: `requests_per_node_total`, `node_active_requests`
- resiliency: `node_failures_total`, `upstream_errors_total`

Grafana:

- [http://localhost:3000](http://localhost:3000)
- dashboard: `grafana/dashboards/inference.json`

## Notes

- If 3 full replicas do not fit one GPU, use fewer nodes or multiple GPUs.
- Every request needs `Authorization: Bearer <AUTH_TOKEN>`.

