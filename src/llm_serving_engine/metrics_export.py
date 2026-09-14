"""Prometheus metrics for the serving engine. `GET /metrics` in server.py exports this
registry via `generate_latest`.

Bucket bounds span 1ms to 60s: fine enough near the decode-latency floor (single-digit
milliseconds per token) while still covering a stalled prefill or a queued-under-load
request at the tail.
"""

from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram
from prometheus_client import generate_latest as _generate_latest

from .metrics import RequestMetrics

__all__ = [
    "CONTENT_TYPE_LATEST",
    "KV_CACHE_UTILIZATION",
    "REGISTRY",
    "generate_latest",
    "record_request",
    "sample_kv_utilization",
]

LATENCY_BUCKETS = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60)

REGISTRY = CollectorRegistry()

REQUESTS_TOTAL = Counter(
    "llm_requests_total",
    "Completed requests, labeled by outcome.",
    labelnames=("outcome",),
    registry=REGISTRY,
)

REQUESTS_IN_FLIGHT = Gauge(
    "llm_requests_in_flight",
    "Requests admitted but not yet finished.",
    registry=REGISTRY,
)

SCHEDULE_LATENCY = Histogram(
    "llm_schedule_latency_seconds",
    "enqueue_time -> admit_time.",
    buckets=LATENCY_BUCKETS,
    registry=REGISTRY,
)

PREFILL_LATENCY = Histogram(
    "llm_prefill_latency_seconds",
    "admit_time -> first_token_time.",
    buckets=LATENCY_BUCKETS,
    registry=REGISTRY,
)

INTER_TOKEN_LATENCY = Histogram(
    "llm_inter_token_latency_seconds",
    "Gaps between successive token_times within a request.",
    buckets=LATENCY_BUCKETS,
    registry=REGISTRY,
)

TOTAL_LATENCY = Histogram(
    "llm_total_latency_seconds",
    "enqueue_time -> done_time.",
    buckets=LATENCY_BUCKETS,
    registry=REGISTRY,
)

KV_CACHE_UTILIZATION = Gauge(
    "llm_kv_cache_utilization",
    "Fraction of KV-cache capacity currently in use, in [0, 1].",
    registry=REGISTRY,
)


def generate_latest() -> bytes:
    """Prometheus text exposition for this module's REGISTRY, not the client library's
    process-global default — server.py's /metrics handler binds to this one so it
    exports the histograms defined here rather than an empty default registry.
    """
    return _generate_latest(REGISTRY)


def sample_kv_utilization(engine) -> float | None:
    """Fraction of KV-cache capacity in use, read from whichever allocator the engine
    holds via its existing public state (no allocator/engine changes needed): block
    count is conserved, so free + in-use blocks always sums to capacity regardless of
    when this is sampled. Duck-typed on the allocator's own public attributes
    (`free_blocks` for BlockAllocator, `used_tokens`/`capacity_tokens` for
    ContiguousAllocator) rather than importing those classes here.
    """
    allocator = getattr(engine, "allocator", None)
    if allocator is None:
        return None
    if hasattr(allocator, "free_blocks"):
        in_use = sum(len(seq.block_table.physical_blocks) for seq in engine.running)
        total = in_use + len(allocator.free_blocks)
        return in_use / total if total else None
    if hasattr(allocator, "used_tokens"):
        capacity = allocator.capacity_tokens
        return allocator.used_tokens / capacity if capacity else None
    return None


def record_request(metrics: RequestMetrics) -> None:
    """Observe one finished request's per-stage latencies. Stages that never happened
    (e.g. a request that failed before admission) are skipped rather than recorded as 0.
    """
    if metrics.schedule_latency is not None:
        SCHEDULE_LATENCY.observe(metrics.schedule_latency)
    if metrics.prefill_latency is not None:
        PREFILL_LATENCY.observe(metrics.prefill_latency)
    for gap in metrics.inter_token_latencies:
        INTER_TOKEN_LATENCY.observe(gap)
    if metrics.total_latency is not None:
        TOTAL_LATENCY.observe(metrics.total_latency)
    REQUESTS_TOTAL.labels(outcome="success" if metrics.done_time is not None else "error").inc()
