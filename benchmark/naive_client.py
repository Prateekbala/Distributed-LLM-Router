import time
from dataclasses import dataclass

import httpx


@dataclass(slots=True)
class RequestResult:
    latency_ms: float
    ttft_ms: float | None
    prompt_tokens: int
    completion_tokens: int
    ok: bool
    node: str | None = None


async def run_naive(
    client: httpx.AsyncClient,
    base_url: str,
    token: str,
    num_requests: int,
    model: str,
    prompt: str,
    max_tokens: int,
    capture_node: bool = False,
) -> list[RequestResult]:
    results: list[RequestResult] = []
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": False,
    }
    for _ in range(num_requests):
        start = time.perf_counter()
        try:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            body = response.json()
            usage = body.get("usage", {})
            node = response.headers.get("x-routed-node") if capture_node else None
            results.append(
                RequestResult(
                    latency_ms=(time.perf_counter() - start) * 1000,
                    ttft_ms=None,
                    prompt_tokens=int(usage.get("prompt_tokens", 0)),
                    completion_tokens=int(usage.get("completion_tokens", 0)),
                    ok=True,
                    node=node,
                )
            )
        except Exception:
            results.append(
                RequestResult(
                    latency_ms=(time.perf_counter() - start) * 1000,
                    ttft_ms=None,
                    prompt_tokens=0,
                    completion_tokens=0,
                    ok=False,
                    node=None,
                )
            )
    return results

