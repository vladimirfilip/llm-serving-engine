"""Benchmark-report aggregation: turns one run's raw per-request result dicts (as
produced by `open_loop_load_gen` + `client.send_request`) into the named metrics a
benchmark report is judged on — TTFT, TPOT, ITL, end-to-end latency, throughput, and
goodput — without recomputing percentile math already in `metrics.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

from ..metrics import LatencySummary, summarize


@dataclass(slots=True)
class SLOThresholds:
    ttft_ms: float | None = None
    tpot_ms: float | None = None
    e2e_ms: float | None = None


@dataclass(slots=True)
class RunReport:
    throughput_req_s: float
    output_tokens_s: float
    input_tokens_s: float
    total_tokens_s: float
    ttft: LatencySummary | None
    tpot: LatencySummary | None
    e2e_latency: LatencySummary | None
    itl: LatencySummary | None
    goodput_req_s: float | None = None
    slo_attainment: dict[str, float] | None = None


def tpot(result: dict) -> float | None:
    ftl, latency, output_tokens = (
        result.get("first_token_latency"),
        result.get("latency"),
        result.get("output_tokens"),
    )
    if ftl is None or latency is None or not output_tokens or output_tokens < 2:
        return None
    return (latency - ftl) / (output_tokens - 1)


def _satisfies(result: dict, slo: SLOThresholds) -> bool:
    checks = (
        (slo.ttft_ms, result.get("first_token_latency")),
        (slo.tpot_ms, tpot(result)),
        (slo.e2e_ms, result.get("latency")),
    )
    return all(threshold_ms is None or (value is not None and value * 1000 <= threshold_ms)
               for threshold_ms, value in checks)


def build_report(results: list[dict], duration_s: float, slo: SLOThresholds | None = None) -> RunReport:
    """`results` is one run's list of per-request dicts. `duration_s` is the run's
    wall-clock length, used as the throughput denominator (not the sum of per-request
    latencies, which would double-count concurrent requests)."""
    successes = [r for r in results if r.get("success", True)]

    ttfts = [r["first_token_latency"] for r in successes if r.get("first_token_latency") is not None]
    e2es = [r["latency"] for r in successes if r.get("latency") is not None]
    tpots = [t for t in (tpot(r) for r in successes) if t is not None]
    itls = [gap for r in successes for gap in _inter_token_latencies(r)]

    output_tokens = sum(r.get("output_tokens") or 0 for r in successes)
    input_tokens = sum(r.get("prompt_tokens") or 0 for r in successes)

    report = RunReport(
        throughput_req_s=len(successes) / duration_s,
        output_tokens_s=output_tokens / duration_s,
        input_tokens_s=input_tokens / duration_s,
        total_tokens_s=(output_tokens + input_tokens) / duration_s,
        ttft=summarize(ttfts) if ttfts else None,
        tpot=summarize(tpots) if tpots else None,
        e2e_latency=summarize(e2es) if e2es else None,
        itl=summarize(itls) if itls else None,
    )
    if slo is not None:
        satisfying = [r for r in successes if _satisfies(r, slo)]
        report.goodput_req_s = len(satisfying) / duration_s
        report.slo_attainment = {
            "ttft": _attainment_pct(successes, lambda r: r.get("first_token_latency"), slo.ttft_ms),
            "tpot": _attainment_pct(successes, tpot, slo.tpot_ms),
            "e2e": _attainment_pct(successes, lambda r: r.get("latency"), slo.e2e_ms),
        }
    return report


def _inter_token_latencies(result: dict) -> list[float]:
    token_times: list[float] = result.get("token_times") or []
    return [b - a for a, b in pairwise(token_times)]


def _attainment_pct(results: list[dict], value_fn, threshold_ms: float | None) -> float | None:
    if threshold_ms is None:
        return None
    values = [value_fn(r) for r in results]
    values = [v for v in values if v is not None]
    if not values:
        return None
    met = sum(1 for v in values if v * 1000 <= threshold_ms)
    return 100.0 * met / len(values)
