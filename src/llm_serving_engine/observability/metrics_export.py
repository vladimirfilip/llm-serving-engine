"""Prometheus metrics for the serving engine, exported by `GET /metrics`.

Bucket bounds span 1ms to 60s: fine near the decode-latency floor (single-digit
milliseconds per token) and wide enough for a stalled prefill or a request queued under
load.
"""

from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram
from prometheus_client import generate_latest as _generate_latest

from .metrics import RequestMetrics

__all__ = [
    "CONTENT_TYPE_LATEST",
    "KV_CACHE_UTILIZATION",
    "PREEMPTIONS_TOTAL",
    "REGISTRY",
    "REQUESTS_IN_FLIGHT",
    "generate_latest",
    "record_request",
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
    "Sequences in `running` after the latest scheduler step.",
    registry=REGISTRY,
)

PREEMPTIONS_TOTAL = Counter(
    "llm_preemptions_total",
    "Running sequences sent back to waiting for lack of KV blocks.",
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
    "Gaps between successive tokens of one request, stamped when the scheduler receives them.",
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
    """Prometheus text exposition of REGISTRY, which holds every metric above."""
    return _generate_latest(REGISTRY)


def record_request(metrics: RequestMetrics) -> None:
    """Observes one ended request's stage latencies. A stage that never happened (an
    aborted request's done_time, say) is skipped; an ended request without done_time
    counts as an error."""
    if metrics.schedule_latency is not None:
        SCHEDULE_LATENCY.observe(metrics.schedule_latency)
    if metrics.prefill_latency is not None:
        PREFILL_LATENCY.observe(metrics.prefill_latency)
    for gap in metrics.inter_token_latencies:
        INTER_TOKEN_LATENCY.observe(gap)
    if metrics.total_latency is not None:
        TOTAL_LATENCY.observe(metrics.total_latency)
    REQUESTS_TOTAL.labels(outcome="success" if metrics.done_time is not None else "error").inc()
