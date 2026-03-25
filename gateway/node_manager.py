"""Per-node state and coordination for the distributed router."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from gateway.metrics import set_node_active_gauge


@dataclass
class NodeState:
    """Runtime stats for one vLLM upstream."""

    url: str
    active_requests: int = 0
    avg_latency: float = 0.0
    latency_samples: int = 0
    error_count: int = 0
    last_updated: float = field(default_factory=time.time)
    healthy: bool = True
    consecutive_health_failures: int = 0

    def record_latency(self, latency_seconds: float) -> None:
        """EMA-style average so recent behavior dominates."""
        self.latency_samples += 1
        alpha = min(0.3, 1.0 / max(self.latency_samples, 1))
        if self.latency_samples == 1:
            self.avg_latency = latency_seconds
        else:
            self.avg_latency = alpha * latency_seconds + (1.0 - alpha) * self.avg_latency
        self.last_updated = time.time()


class NodeManager:
    """Thread-safe node registry with load and health bookkeeping."""

    def __init__(self, urls: list[str]) -> None:
        self._nodes: dict[str, NodeState] = {u.rstrip("/"): NodeState(url=u.rstrip("/")) for u in urls}
        self._lock = asyncio.Lock()

    def all_urls(self) -> list[str]:
        return list(self._nodes.keys())

    async def get_node(self, url: str) -> NodeState | None:
        key = url.rstrip("/")
        async with self._lock:
            return self._nodes.get(key)

    async def get_available_nodes(self, exclude: frozenset[str] | None = None) -> list[NodeState]:
        ex = exclude or frozenset()
        async with self._lock:
            return [n for n in self._nodes.values() if n.healthy and n.url not in ex]

    async def mark_request_start(self, node: NodeState) -> None:
        async with self._lock:
            node.active_requests += 1
            node.last_updated = time.time()
            set_node_active_gauge(node.url, node.active_requests)

    async def mark_request_end(self, node: NodeState, latency_seconds: float | None, success: bool) -> None:
        async with self._lock:
            node.active_requests = max(0, node.active_requests - 1)
            if success and latency_seconds is not None:
                node.record_latency(latency_seconds)
            node.last_updated = time.time()
            set_node_active_gauge(node.url, node.active_requests)

    async def mark_failure(self, node: NodeState) -> None:
        async with self._lock:
            node.error_count += 1
            node.last_updated = time.time()

    async def mark_health_success(self, node: NodeState) -> None:
        async with self._lock:
            node.consecutive_health_failures = 0
            node.healthy = True
            node.last_updated = time.time()

    async def mark_health_failure(self, node: NodeState) -> None:
        async with self._lock:
            node.consecutive_health_failures += 1
            node.last_updated = time.time()

    async def set_unhealthy(self, node: NodeState, threshold: int) -> bool:
        """Returns True if node was marked unhealthy."""
        async with self._lock:
            if node.consecutive_health_failures >= threshold:
                node.healthy = False
                return True
            return False

    async def get_node_stats(self) -> list[dict[str, float | int | str]]:
        async with self._lock:
            out: list[dict[str, float | int | str]] = []
            for n in self._nodes.values():
                out.append(
                    {
                        "url": n.url,
                        "active_requests": n.active_requests,
                        "avg_latency": round(n.avg_latency, 6),
                        "errors": n.error_count,
                        "healthy": n.healthy,
                    }
                )
            return out
