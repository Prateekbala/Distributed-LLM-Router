import asyncio
import statistics
import time
from dataclasses import dataclass

import httpx

from benchmark.naive_client import RequestResult


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * p
    f = int(k)
    c = min(f + 1, len(ordered) - 1)
    if f == c:
        return ordered[f]
    return ordered[f] + (ordered[c] - ordered[f]) * (k - f)


@dataclass(slots=True)
class AggregateMetrics:
    total_time_seconds: float
    requests_per_second: float
    tokens_per_second: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    time_to_first_token_p50_ms: float
    error_rate: float


async def run_concurrent(
    client: httpx.AsyncClient,
    base_url: str,
    token: str,
    num_requests: int,
    concurrency: int,
    model: str,
    prompt: str,
    max_tokens: int,
    capture_node: bool = False,
) -> tuple[list[RequestResult], AggregateMetrics]:
    sem = asyncio.Semaphore(concurrency)
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": False,
    }

    async def worker() -> RequestResult:
        async with sem:
            start = time.perf_counter()
            try:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                body = response.json()
                usage = body.get("usage", {})
                node = response.headers.get("x-routed-node") if capture_node else None
                return RequestResult(
                    latency_ms=(time.perf_counter() - start) * 1000,
                    ttft_ms=None,
                    prompt_tokens=int(usage.get("prompt_tokens", 0)),
                    completion_tokens=int(usage.get("completion_tokens", 0)),
                    ok=True,
                    node=node,
                )
            except Exception:
                return RequestResult(
                    latency_ms=(time.perf_counter() - start) * 1000,
                    ttft_ms=None,
                    prompt_tokens=0,
                    completion_tokens=0,
                    ok=False,
                    node=None,
                )

    started = time.perf_counter()
    results = await asyncio.gather(*(worker() for _ in range(num_requests)))
    total_time = max(time.perf_counter() - started, 1e-9)

    latencies = [r.latency_ms for r in results]
    ttfts = [r.ttft_ms for r in results if r.ttft_ms is not None]
    total_completion_tokens = sum(r.completion_tokens for r in results)
    errors = len([r for r in results if not r.ok])
    metrics = AggregateMetrics(
        total_time_seconds=total_time,
        requests_per_second=num_requests / total_time,
        tokens_per_second=total_completion_tokens / total_time,
        latency_p50_ms=statistics.median(latencies) if latencies else 0.0,
        latency_p95_ms=percentile(latencies, 0.95),
        latency_p99_ms=percentile(latencies, 0.99),
        time_to_first_token_p50_ms=statistics.median(ttfts) if ttfts else 0.0,
        error_rate=(errors / max(num_requests, 1)) * 100,
    )
    return results, metrics

