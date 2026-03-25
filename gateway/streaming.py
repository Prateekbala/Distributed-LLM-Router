"""SSE streaming from a single upstream node."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator

import httpx

from gateway.chat_utils import extract_usage
from gateway.metrics import (
    RequestMetricsInput,
    active_requests,
    observe_node_failure,
    observe_node_success,
    observe_request,
)
from gateway.node_manager import NodeManager, NodeState


async def stream_upstream_response(
    upstream_response: httpx.Response,
    model: str,
    started_gateway: float,
    node_upstream_started: float,
    node: NodeState,
    manager: NodeManager,
    semaphore: asyncio.Semaphore,
) -> AsyncIterator[bytes]:
    first_chunk_at: float | None = None
    prompt_tokens = 0
    completion_tokens = 0
    success = False
    try:
        async for chunk in upstream_response.aiter_bytes():
            if chunk and first_chunk_at is None:
                first_chunk_at = time.perf_counter()
            if chunk:
                for line in chunk.decode("utf-8", errors="ignore").splitlines():
                    if not line.startswith("data: "):
                        continue
                    event = line.removeprefix("data: ").strip()
                    if event == "[DONE]":
                        continue
                    try:
                        payload = json.loads(event)
                    except json.JSONDecodeError:
                        continue
                    p, c = extract_usage(payload)
                    prompt_tokens = max(prompt_tokens, p)
                    completion_tokens = max(completion_tokens, c)
            yield chunk
        success = True
    finally:
        await upstream_response.aclose()
        gateway_elapsed = time.perf_counter() - started_gateway
        node_elapsed = time.perf_counter() - node_upstream_started
        ttft = (first_chunk_at - started_gateway) if first_chunk_at else None
        await manager.mark_request_end(node, node_elapsed, success)
        if success:
            observe_node_success(node.url, node_elapsed)
        else:
            observe_node_failure(node.url)
            await manager.mark_failure(node)
        observe_request(
            RequestMetricsInput(
                model=model,
                success=success,
                latency_seconds=gateway_elapsed,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                ttft_seconds=ttft,
            )
        )
        active_requests.dec()
        semaphore.release()
