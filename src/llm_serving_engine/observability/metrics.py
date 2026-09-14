"""Per-request latency decomposition and cross-request aggregation.

RequestMetrics instruments every stage boundary of a request's lifetime so a bad
tail latency can be attributed to a stage, not guessed at.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(slots=True)
class RequestMetrics:
    enqueue_time: float
    admit_time: float | None = None
    first_token_time: float | None = None
    token_times: list[float] = field(default_factory=list)
    done_time: float | None = None

    @property
    def schedule_latency(self) -> float | None:
        if self.admit_time is None:
            return None
        return self.admit_time - self.enqueue_time

    @property
    def prefill_latency(self) -> float | None:
        if self.first_token_time is None or self.admit_time is None:
            return None
        return self.first_token_time - self.admit_time

    @property
    def inter_token_latencies(self) -> list[float]:
        return [b - a for a, b in zip(self.token_times, self.token_times[1:], strict=False)]

    @property
    def total_latency(self) -> float | None:
        if self.done_time is None:
            return None
        return self.done_time - self.enqueue_time


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile, p in [0, 100]. Empty input is a caller error."""
    if not values:
        raise ValueError("percentile of empty sequence")
    ordered = sorted(values)
    rank = math.ceil(p / 100 * len(ordered)) - 1
    return ordered[max(0, min(rank, len(ordered) - 1))]


@dataclass
class LatencySummary:
    count: int
    p50: float
    p90: float
    p95: float
    p99: float
    p999: float
    mean: float
    min: float
    max: float


def summarize(values: list[float]) -> LatencySummary:
    if not values:
        raise ValueError("summarize of empty sequence")
    return LatencySummary(
        count=len(values),
        p50=percentile(values, 50),
        p90=percentile(values, 90),
        p95=percentile(values, 95),
        p99=percentile(values, 99),
        p999=percentile(values, 99.9),
        mean=sum(values) / len(values),
        min=min(values),
        max=max(values),
    )


def summarize_stage_latencies(metrics: list[RequestMetrics]) -> dict[str, LatencySummary]:
    """One LatencySummary per named stage, over whichever requests have that stage set."""
    stages: dict[str, list[float]] = {
        "schedule_latency": [],
        "prefill_latency": [],
        "total_latency": [],
        "inter_token_latency": [],
    }
    for m in metrics:
        if m.schedule_latency is not None:
            stages["schedule_latency"].append(m.schedule_latency)
        if m.prefill_latency is not None:
            stages["prefill_latency"].append(m.prefill_latency)
        if m.total_latency is not None:
            stages["total_latency"].append(m.total_latency)
        stages["inter_token_latency"].extend(m.inter_token_latencies)
    return {name: summarize(vals) for name, vals in stages.items() if vals}
