from dataclasses import dataclass

from prometheus_client import Counter, Gauge, Histogram, generate_latest

LATENCY_BUCKETS = (0.1, 0.5, 1, 2, 5, 10, 30)

requests_total = Counter(
    "requests_total",
    "Total requests handled",
    labelnames=("status", "model"),
)
tokens_generated_total = Counter(
    "tokens_generated_total",
    "Total generated tokens",
    labelnames=("model",),
)
tokens_prompt_total = Counter(
    "tokens_prompt_total",
    "Total prompt tokens",
    labelnames=("model",),
)
requests_rejected_total = Counter(
    "requests_rejected_total",
    "Total rejected requests",
    labelnames=("reason",),
)
upstream_errors_total = Counter(
    "upstream_errors_total",
    "Total upstream errors",
    labelnames=("code",),
)
request_latency_seconds = Histogram(
    "request_latency_seconds",
    "End-to-end request latency in seconds",
    buckets=LATENCY_BUCKETS,
)
time_to_first_token_seconds = Histogram(
    "time_to_first_token_seconds",
    "Time to first token in seconds",
    buckets=LATENCY_BUCKETS,
)
tokens_per_second = Histogram(
    "tokens_per_second",
    "Token throughput distribution",
    buckets=LATENCY_BUCKETS,
)
active_requests = Gauge("active_requests", "Current active requests")
queue_depth = Gauge("queue_depth", "Current queued requests")
queue_capacity = Gauge("queue_capacity", "Configured queue size")

requests_per_node_total = Counter(
    "requests_per_node_total",
    "Total requests routed to each upstream node",
    labelnames=("node",),
)
node_latency_seconds = Histogram(
    "node_latency_seconds",
    "Observed upstream latency per node in seconds",
    labelnames=("node",),
    buckets=LATENCY_BUCKETS,
)
node_failures_total = Counter(
    "node_failures_total",
    "Failures attributed to each upstream node",
    labelnames=("node",),
)
node_active_requests = Gauge(
    "node_active_requests",
    "Current active requests per upstream node",
    labelnames=("node",),
)


def set_node_active_gauge(node_url: str, value: int) -> None:
    node_active_requests.labels(node=node_url).set(value)


def observe_node_success(node_url: str, latency_seconds: float) -> None:
    requests_per_node_total.labels(node=node_url).inc()
    node_latency_seconds.labels(node=node_url).observe(latency_seconds)


def observe_node_failure(node_url: str) -> None:
    node_failures_total.labels(node=node_url).inc()


@dataclass(slots=True)
class RequestMetricsInput:
    model: str
    success: bool
    latency_seconds: float
    prompt_tokens: int
    completion_tokens: int
    ttft_seconds: float | None = None


def observe_request(data: RequestMetricsInput) -> None:
    requests_total.labels(status="success" if data.success else "error", model=data.model).inc()
    request_latency_seconds.observe(data.latency_seconds)
    if data.prompt_tokens > 0:
        tokens_prompt_total.labels(model=data.model).inc(data.prompt_tokens)
    if data.completion_tokens > 0:
        tokens_generated_total.labels(model=data.model).inc(data.completion_tokens)
        tps = data.completion_tokens / data.latency_seconds if data.latency_seconds > 0 else 0.0
        if tps > 0:
            tokens_per_second.observe(tps)
    if data.ttft_seconds is not None and data.ttft_seconds >= 0:
        time_to_first_token_seconds.observe(data.ttft_seconds)


def metrics_payload() -> bytes:
    return generate_latest()
