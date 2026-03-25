import asyncio
import logging
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from gateway.config import Settings, get_settings
from gateway.health import health_check_loop
from gateway.metrics import (
    active_requests,
    metrics_payload,
    queue_capacity,
    queue_depth,
    requests_rejected_total,
    set_node_active_gauge,
)
from gateway.middleware import LoggingMiddleware, RequestIDMiddleware, validate_bearer_auth
from gateway.node_manager import NodeManager
from gateway.proxy_chat import forward_chat_completions
from gateway.router import Router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("gateway.main")


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str | None = None
    stream: bool = False
    max_tokens: int | None = Field(default=None, ge=1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    timeout = httpx.Timeout(settings.node_timeout_seconds, connect=settings.connect_timeout_seconds)
    app.state.client = httpx.AsyncClient(timeout=timeout)
    app.state.semaphore = asyncio.Semaphore(settings.max_concurrent_requests)
    app.state.waiting = 0
    app.state.node_manager = NodeManager(settings.vllm_nodes)
    app.state.router = Router(app.state.node_manager, settings.routing_strategy)
    queue_capacity.set(settings.max_queue_size)
    for url in app.state.node_manager.all_urls():
        set_node_active_gauge(url, 0)
    health_task = asyncio.create_task(health_check_loop(app))
    app.state.health_task = health_task
    try:
        yield
    finally:
        health_task.cancel()
        try:
            await health_task
        except asyncio.CancelledError:
            pass
        await app.state.client.aclose()


app = FastAPI(title="LLM Gateway", version="1.0.0", lifespan=lifespan)
app.add_middleware(RequestIDMiddleware)
app.add_middleware(LoggingMiddleware)


@app.get("/health")
async def health(settings: Settings = Depends(get_settings)) -> dict:
    return {
        "status": "ok",
        "model": settings.model_name,
        "upstream_nodes": settings.vllm_nodes,
        "routing_strategy": settings.routing_strategy,
    }


@app.get("/metrics")
async def metrics() -> Response:
    return Response(content=metrics_payload(), media_type="text/plain; version=0.0.4")


@app.get("/stats")
async def stats() -> dict:
    mgr: NodeManager = app.state.node_manager
    nodes = await mgr.get_node_stats()
    return {
        "active_requests": active_requests._value.get(),
        "queue_depth": queue_depth._value.get(),
        "nodes": nodes,
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, body: ChatCompletionRequest, settings: Settings = Depends(get_settings)):
    validate_bearer_auth(request, settings)
    if app.state.semaphore.locked():
        if app.state.waiting >= settings.max_queue_size:
            requests_rejected_total.labels(reason="queue_full").inc()
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Gateway queue full")
        app.state.waiting += 1
        queue_depth.set(app.state.waiting)

    await app.state.semaphore.acquire()
    if app.state.waiting > 0:
        app.state.waiting -= 1
        queue_depth.set(app.state.waiting)
    active_requests.inc()
    started = time.perf_counter()
    model = body.model or settings.model_name
    payload = body.model_dump(exclude_none=True)
    payload["model"] = model
    stream_handoff = False
    try:
        result = await forward_chat_completions(
            request=request,
            payload=payload,
            model=model,
            stream=body.stream,
            settings=settings,
            client=app.state.client,
            semaphore=app.state.semaphore,
            manager=app.state.node_manager,
            router=app.state.router,
            started_gateway=started,
        )
        if isinstance(result, StreamingResponse):
            stream_handoff = True
        return result
    finally:
        if not stream_handoff:
            active_requests.dec()
            app.state.semaphore.release()

