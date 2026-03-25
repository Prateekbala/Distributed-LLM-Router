"""Routed upstream proxy for /v1/chat/completions with retries and streaming."""

from __future__ import annotations

import asyncio
import logging
import time

import httpx
from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse

from gateway.chat_utils import extract_usage
from gateway.config import Settings
from gateway.metrics import (
    RequestMetricsInput,
    observe_node_failure,
    observe_node_success,
    observe_request,
    upstream_errors_total,
)
from gateway.node_manager import NodeManager
from gateway.router import Router
from gateway.streaming import stream_upstream_response

logger = logging.getLogger("gateway.proxy")


def _should_retry(exc: BaseException) -> bool:
    if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500
    return False


async def forward_chat_completions(
    request: Request,
    payload: dict,
    model: str,
    stream: bool,
    settings: Settings,
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    manager: NodeManager,
    router: Router,
    started_gateway: float,
) -> JSONResponse | StreamingResponse:
    headers = {
        "content-type": "application/json",
        "authorization": request.headers.get("authorization", ""),
        "x-request-id": getattr(request.state, "request_id", "unknown"),
    }
    failed: set[str] = set()
    max_attempts = settings.max_retries + 1
    last_error: BaseException | None = None

    for attempt in range(max_attempts):
        node = await router.select_node({}, exclude=frozenset(failed))
        if node is None:
            break
        await manager.mark_request_start(node)
        node_attempt_started = time.perf_counter()
        endpoint = f"{node.url}/v1/chat/completions"
        try:
            if stream:
                upstream = client.build_request("POST", endpoint, headers=headers, json=payload)
                response = await client.send(upstream, stream=True)
                response.raise_for_status()
                return StreamingResponse(
                    stream_upstream_response(
                        response,
                        model=model,
                        started_gateway=started_gateway,
                        node_upstream_started=node_attempt_started,
                        node=node,
                        manager=manager,
                        semaphore=semaphore,
                    ),
                    media_type="text/event-stream",
                    headers={"X-Routed-Node": node.url},
                )

            response = await client.post(endpoint, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            elapsed = time.perf_counter() - node_attempt_started
            await manager.mark_request_end(node, elapsed, True)
            observe_node_success(node.url, elapsed)
            prompt_tokens, completion_tokens = extract_usage(result)
            observe_request(
                RequestMetricsInput(
                    model=model,
                    success=True,
                    latency_seconds=time.perf_counter() - started_gateway,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    ttft_seconds=None,
                )
            )
            return JSONResponse(content=result, headers={"X-Routed-Node": node.url})
        except (
            httpx.TimeoutException,
            httpx.ConnectError,
            httpx.HTTPStatusError,
        ) as exc:
            last_error = exc
            await manager.mark_request_end(node, None, False)
            await manager.mark_failure(node)
            observe_node_failure(node.url)
            failed.add(node.url)
            if isinstance(exc, httpx.TimeoutException):
                upstream_errors_total.labels(code="timeout").inc()
            elif isinstance(exc, httpx.ConnectError):
                upstream_errors_total.labels(code="connection").inc()
            else:
                upstream_errors_total.labels(code="bad_gateway").inc()
            if attempt + 1 < max_attempts and _should_retry(exc):
                await asyncio.sleep(min(0.1 * (2**attempt), 2.0))
                continue
            observe_request(
                RequestMetricsInput(
                    model=model,
                    success=False,
                    latency_seconds=time.perf_counter() - started_gateway,
                    prompt_tokens=0,
                    completion_tokens=0,
                )
            )
            _raise_upstream_error(exc)
        except Exception as exc:
            last_error = exc
            await manager.mark_request_end(node, None, False)
            await manager.mark_failure(node)
            observe_node_failure(node.url)
            failed.add(node.url)
            logger.exception("upstream_unexpected_error", extra={"node": node.url})
            observe_request(
                RequestMetricsInput(
                    model=model,
                    success=False,
                    latency_seconds=time.perf_counter() - started_gateway,
                    prompt_tokens=0,
                    completion_tokens=0,
                )
            )
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Upstream error",
            ) from exc

    observe_request(
        RequestMetricsInput(
            model=model,
            success=False,
            latency_seconds=time.perf_counter() - started_gateway,
            prompt_tokens=0,
            completion_tokens=0,
        )
    )
    if last_error is not None:
        _raise_upstream_error(last_error)
    raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="No healthy upstream nodes")


def _raise_upstream_error(exc: BaseException) -> None:
    if isinstance(exc, httpx.TimeoutException):
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail="Upstream timeout") from exc
    if isinstance(exc, httpx.ConnectError):
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Upstream unavailable") from exc
    if isinstance(exc, httpx.HTTPStatusError):
        detail = {"error": "Upstream returned error", "status_code": exc.response.status_code}
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=detail) from exc
    raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Upstream error") from exc
