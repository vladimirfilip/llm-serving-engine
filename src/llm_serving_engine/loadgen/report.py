"""Benchmark-report aggregation: one run's raw per-request results (from a timing loop and
`client.send_request`) into the metrics a benchmark report is judged on: TTFT, TPOT, ITL,
end-to-end latency, throughput and goodput."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from statistics import median

from ..observability.metrics import LatencySummary, summarize

# An open-loop run keeps up when late arrivals wait no longer than this multiple of early ones.
MAX_KEEPING_UP_LATENCY_GROWTH = 2.0


@dataclass(slots=True)
class SLOThresholds:
    ttft_ms: float | None = None
    tpot_ms: float | None = None
    e2e_ms: float | None = None


@dataclass(slots=True)
class RunReport:
    wall_clock_s: float
    failures: int
    throughput_req_s: float
    output_tokens_s: float
    input_tokens_s: float
    total_tokens_s: float
    # TTFT grows with prompt length, so it is summarized per request shape, never blended.
    # Open loop only: a closed loop's TTFT measures the queue its own clients hold full.
    ttft_by_shape: dict[str, LatencySummary]
    tpot: LatencySummary | None
    e2e_latency: LatencySummary | None
    itl: LatencySummary | None
    latency_growth: float | None = None
    goodput_req_s: float | None = None
    slo_attainment: dict[str, float] | None = None

    @property
    def keeps_up(self) -> bool:
        """No request failed and the backlog didn't grow over the run (open loop only)."""
        growth = self.latency_growth
        return self.failures == 0 and (growth is None or growth <= MAX_KEEPING_UP_LATENCY_GROWTH)


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


def latency_growth(results: list[dict]) -> float | None:
    """Median latency of the last quarter of scheduled arrivals over the first quarter's.
    A server that keeps up holds it near 1; a growing backlog pushes it up. None for a
    closed-loop run, or one too short to have quarters."""
    scheduled = sorted(
        (r for r in results if "scheduled_at" in r and r.get("success", True)),
        key=lambda r: r["scheduled_at"],
    )
    quarter = len(scheduled) // 4
    if quarter == 0:
        return None
    first = median(r["latency"] for r in scheduled[:quarter])
    last = median(r["latency"] for r in scheduled[-quarter:])
    return last / first


def build_report(results: list[dict], slo: SLOThresholds | None = None) -> RunReport:
    successes = [r for r in results if r.get("success", True)]
    elapsed = wall_clock_s(results)

    def per_second(count: float) -> float:
        return count / elapsed if elapsed else 0.0

    ttfts_by_shape: dict[str, list[float]] = {}
    for r in successes:
        if "scheduled_at" in r and r["first_token_latency"] is not None:
            ttfts_by_shape.setdefault(r["shape"], []).append(r["first_token_latency"])
    e2es = [r["latency"] for r in successes]
    tpots = [t for t in (tpot(r) for r in successes) if t is not None]
    itls = [b - a for r in successes for a, b in pairwise(r.get("token_times") or [])]
    output_tokens = sum(r.get("output_tokens") or 0 for r in successes)
    input_tokens = sum(r.get("prompt_tokens") or 0 for r in successes)

    report = RunReport(
        wall_clock_s=elapsed,
        failures=len(results) - len(successes),
        throughput_req_s=per_second(len(successes)),
        output_tokens_s=per_second(output_tokens),
        input_tokens_s=per_second(input_tokens),
        total_tokens_s=per_second(output_tokens + input_tokens),
        ttft_by_shape={shape: summarize(ttfts) for shape, ttfts in sorted(ttfts_by_shape.items())},
        tpot=summarize(tpots) if tpots else None,
        e2e_latency=summarize(e2es) if e2es else None,
        itl=summarize(itls) if itls else None,
        latency_growth=latency_growth(results),
    )
    if slo is not None:
        satisfying = [r for r in successes if _satisfies(r, slo)]
        report.goodput_req_s = per_second(len(satisfying))
        report.slo_attainment = {
            "ttft": _attainment_pct(successes, lambda r: r.get("first_token_latency"), slo.ttft_ms),
            "tpot": _attainment_pct(successes, tpot, slo.tpot_ms),
            "e2e": _attainment_pct(successes, lambda r: r.get("latency"), slo.e2e_ms),
        }
    return report


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
