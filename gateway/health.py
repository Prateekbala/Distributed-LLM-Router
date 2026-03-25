"""Periodic upstream health probes."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

logger = logging.getLogger("gateway.health")


async def health_check_loop(app: Any) -> None:
    from gateway.config import get_settings

    settings = get_settings()
    interval = max(1.0, settings.health_check_interval_seconds)
    threshold = max(1, settings.health_failure_threshold)

    while True:
        await asyncio.sleep(interval)
        client: httpx.AsyncClient = app.state.client
        manager = app.state.node_manager
        for url in manager.all_urls():
            node = await manager.get_node(url)
            if node is None:
                continue
            ok = await _probe(client, url)
            if ok:
                await manager.mark_health_success(node)
            else:
                await manager.mark_health_failure(node)
                marked = await manager.set_unhealthy(node, threshold)
                if marked:
                    logger.warning("node_marked_unhealthy", extra={"url": url})


async def _probe(client: httpx.AsyncClient, base_url: str) -> bool:
    probe_url = f"{base_url.rstrip('/')}/v1/models"
    try:
        response = await client.get(probe_url, timeout=httpx.Timeout(5.0, connect=2.0))
        return response.status_code < 500
    except httpx.HTTPError:
        return False
