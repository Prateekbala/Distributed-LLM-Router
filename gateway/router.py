"""Routing strategies for selecting upstream vLLM nodes."""

from __future__ import annotations

import asyncio
from typing import Any

from gateway.config import RoutingStrategy
from gateway.node_manager import NodeManager, NodeState


class Router:
    """Selects an upstream node according to the configured strategy."""

    def __init__(self, manager: NodeManager, strategy: RoutingStrategy) -> None:
        self._manager = manager
        self._strategy = strategy
        self._rr_lock = asyncio.Lock()
        self._rr_index = 0

    async def select_node(
        self,
        request_context: dict[str, Any],
        exclude: frozenset[str],
    ) -> NodeState | None:
        nodes = await self._manager.get_available_nodes(exclude=exclude)
        if not nodes:
            return None
        if self._strategy == "round_robin":
            return await self._round_robin(nodes)
        if self._strategy == "least_loaded":
            return self._least_loaded(nodes)
        return self._latency_based(nodes)

    async def _round_robin(self, nodes: list[NodeState]) -> NodeState:
        ordered = sorted(nodes, key=lambda n: n.url)
        async with self._rr_lock:
            idx = self._rr_index % len(ordered)
            self._rr_index += 1
            return ordered[idx]

    def _least_loaded(self, nodes: list[NodeState]) -> NodeState:
        return min(nodes, key=lambda n: (n.active_requests, n.url))

    def _latency_based(self, nodes: list[NodeState]) -> NodeState:
        def key(n: NodeState) -> tuple[float, int, str]:
            if n.latency_samples == 0:
                return (float("inf"), n.active_requests, n.url)
            return (n.avg_latency, n.active_requests, n.url)

        return min(nodes, key=key)
