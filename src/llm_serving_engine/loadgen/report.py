"""Benchmark-report aggregation: one run's raw per-request results (from a timing loop and
`client.send_request`) into the metrics a benchmark report is judged on: TTFT, TPOT, ITL,
end-to-end latency, throughput and goodput."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import pairwise
from statistics import median

from ..observability.metrics import LatencySummary, summarize

# An open-loop run keeps up when late arrivals wait no longer than this multiple of early ones.
MAX_KEEPING_UP_TTFT_GROWTH = 2.0
# Rates count completions from this fraction of the run to its end. Before it, requests are
# still filling the server, so completions lag arrivals; after the run, the backlog drains
# below the load being measured. In between, completions track arrivals when the server keeps
# up, and its capacity when it doesn't.
STEADY_START_FRACTION = 0.25
# Below this many requests per quarter, a quarter's median TTFT hinges on which few requests
# landed in it, and the growth ratio swings either way on an idle server.
MIN_QUARTER_REQUESTS = 10


@dataclass(slots=True)
class SLOThresholds:
    ttft_ms: float | None = None
    tpot_ms: float | None = None
    e2e_ms: float | None = None


@dataclass(slots=True)
class LatencyReport:
    # TTFT grows with prompt length and end-to-end latency with output length, so both are
    # summarized per request shape, never blended. TTFT is open loop only: a closed loop's
    # TTFT measures the queue its own clients hold full.
    ttft_by_shape: dict[str, LatencySummary]
    e2e_by_shape: dict[str, LatencySummary]
    e2e_latency: LatencySummary | None
    tpot: LatencySummary | None
    itl: LatencySummary | None


@dataclass(slots=True)
class RunReport:
    duration_s: float
    wall_clock_s: float
    requests: int
    failures: int
    # Open loop only: arrivals actually drawn over the schedule, which for a short Poisson
    # run can sit well off the target rate.
    offered_req_s: float | None
    throughput_req_s: float
    output_tokens_s: float
    input_tokens_s: float
    total_tokens_s: float
    latency: LatencyReport
    ttft_growth: float | None = None
    goodput_req_s: float | None = None
    slo_attainment: dict[str, float | None] | None = None

    @property
    def keeps_up(self) -> bool | None:
        """Open loop only: False if a request failed or TTFT grew past the limit over the run,
        None for a closed loop or a run too short to tell."""
        if self.offered_req_s is None:
            return None
        if self.failures:
            return False
        if self.ttft_growth is None:
            return None
        return self.ttft_growth <= MAX_KEEPING_UP_TTFT_GROWTH


@dataclass(slots=True)
class PointReport:
    """Repeated runs of one config at one load. Rates keep one value per repeat, so their
    spread shows; latencies pool every repeat's requests, so tails rest on more samples."""

    repeats: list[RunReport]
    latency: LatencyReport

    @property
    def keeps_up(self) -> bool | None:
        verdicts = [r.keeps_up for r in self.repeats]
        if False in verdicts:
            return False
        return True if True in verdicts else None


def tpot(result: dict) -> float | None:
    ftl, latency, output_tokens = (
        result.get("first_token_latency"),
        result.get("latency"),
        result.get("output_tokens"),
    )
    if ftl is None or latency is None or not output_tokens or output_tokens < 2:
        return None
    return (latency - ftl) / (output_tokens - 1)


def wall_clock_s(results: list[dict]) -> float:
    """From the loop's start to its last completion."""
    return max((r["completed_at"] for r in results), default=0.0)


def ttft_growth(results: list[dict]) -> float | None:
    """Median TTFT of the last quarter of scheduled arrivals over the first quarter's, for the
    run's most common request shape: queueing shows first in TTFT, and one shape keeps the mix
    of prompt lengths out of the ratio. Near 1 when the server keeps up; a growing backlog
    pushes it up. None for a closed loop, or quarters under MIN_QUARTER_REQUESTS."""
    timed = [
        r for r in results
        if "scheduled_at" in r and r.get("success", True) and r["first_token_latency"] is not None
    ]
    if not timed:
        return None
    [(common, _count)] = Counter(r["shape"] for r in timed).most_common(1)
    scheduled = sorted((r for r in timed if r["shape"] == common), key=lambda r: r["scheduled_at"])
    quarter = len(scheduled) // 4
    if quarter < MIN_QUARTER_REQUESTS:
        return None
    first = median(r["first_token_latency"] for r in scheduled[:quarter])
    last = median(r["first_token_latency"] for r in scheduled[-quarter:])
    return last / first


def summarize_latencies(results: list[dict]) -> LatencyReport:
    successes = [r for r in results if r.get("success", True)]
    ttfts_by_shape: dict[str, list[float]] = {}
    e2es_by_shape: dict[str, list[float]] = {}
    for r in successes:
        e2es_by_shape.setdefault(r["shape"], []).append(r["latency"])
        if "scheduled_at" in r and r["first_token_latency"] is not None:
            ttfts_by_shape.setdefault(r["shape"], []).append(r["first_token_latency"])
    e2es = [r["latency"] for r in successes]
    tpots = [t for t in (tpot(r) for r in successes) if t is not None]
    itls = [b - a for r in successes for a, b in pairwise(r.get("token_times") or [])]
    return LatencyReport(
        ttft_by_shape={shape: summarize(v) for shape, v in sorted(ttfts_by_shape.items())},
        e2e_by_shape={shape: summarize(v) for shape, v in sorted(e2es_by_shape.items())},
        e2e_latency=summarize(e2es) if e2es else None,
        tpot=summarize(tpots) if tpots else None,
        itl=summarize(itls) if itls else None,
    )


def build_report(
    results: list[dict], duration_s: float, slo: SLOThresholds | None = None
) -> RunReport:
    """`duration_s` is how long the loop sent requests. Rates count the successes completed
    in the steady window, from STEADY_START_FRACTION of `duration_s` to its end."""
    open_loop = any("scheduled_at" in r for r in results)
    successes = [r for r in results if r.get("success", True)]
    steady_start = STEADY_START_FRACTION * duration_s
    counted = [r for r in successes if steady_start <= r["completed_at"] <= duration_s]
    elapsed = duration_s - steady_start

    def per_second(count: float) -> float:
        return count / elapsed

    output_tokens = sum(r.get("output_tokens") or 0 for r in counted)
    input_tokens = sum(r.get("prompt_tokens") or 0 for r in counted)
    report = RunReport(
        duration_s=duration_s,
        wall_clock_s=wall_clock_s(results),
        requests=len(results),
        failures=len(results) - len(successes),
        offered_req_s=len(results) / duration_s if open_loop else None,
        throughput_req_s=per_second(len(counted)),
        output_tokens_s=per_second(output_tokens),
        input_tokens_s=per_second(input_tokens),
        total_tokens_s=per_second(output_tokens + input_tokens),
        latency=summarize_latencies(results),
        ttft_growth=ttft_growth(results),
    )
    if slo is not None:
        report.goodput_req_s = per_second(sum(1 for r in counted if _satisfies(r, slo)))
        report.slo_attainment = {
            "ttft": _attainment_pct(successes, lambda r: r.get("first_token_latency"), slo.ttft_ms),
            "tpot": _attainment_pct(successes, tpot, slo.tpot_ms),
            "e2e": _attainment_pct(successes, lambda r: r.get("latency"), slo.e2e_ms),
        }
    return report


def build_point(
    runs: list[list[dict]], duration_s: float, slo: SLOThresholds | None = None
) -> PointReport:
    return PointReport(
        repeats=[build_report(results, duration_s, slo) for results in runs],
        latency=summarize_latencies([r for results in runs for r in results]),
    )


def _satisfies(result: dict, slo: SLOThresholds) -> bool:
    checks = (
        (slo.ttft_ms, result.get("first_token_latency")),
        (slo.tpot_ms, tpot(result)),
        (slo.e2e_ms, result.get("latency")),
    )
    return all(
        threshold_ms is None or (value is not None and value * 1000 <= threshold_ms)
        for threshold_ms, value in checks
    )


def _attainment_pct(results: list[dict], value_fn, threshold_ms: float | None) -> float | None:
    if threshold_ms is None:
        return None
    values = [v for v in (value_fn(r) for r in results) if v is not None]
    if not values:
        return None
    return 100.0 * sum(1 for v in values if v * 1000 <= threshold_ms) / len(values)
