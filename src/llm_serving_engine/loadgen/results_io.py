"""Writes a run's raw per-request results and aggregate report to disk in both JSON
and CSV, so plots can be regenerated from either without rerunning the benchmark.

File names append to a stem rather than replace its suffix: a stem like `qps_0.03` already
contains a dot, and `Path.with_suffix` would read ".03" as the suffix to replace.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

from .report import LatencyReport, RunReport, tpot

RAW_FIELDS = [
    "shape", "success", "error", "scheduled_at_s", "completed_at_s", "latency_s",
    "first_token_latency_s", "tpot_s", "prompt_tokens", "output_tokens", "num_tokens_received",
]


def sibling(stem: Path, suffix: str) -> Path:
    return stem.with_name(stem.name + suffix)


def write_run(stem: Path, run: dict, report: RunReport, **fields) -> None:
    """`run` verbatim to <stem>.json and one row per request to <stem>.csv; `report`, with
    run-identifying `fields` such as `target_qps`, to <stem>_summary.json/.csv."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    sibling(stem, ".json").write_text(json.dumps(run, indent=2))
    with sibling(stem, ".csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_FIELDS)
        writer.writeheader()
        for r in run["results"]:
            writer.writerow({
                "shape": r.get("shape"),
                "success": r.get("success", True),
                "error": r.get("error"),
                "scheduled_at_s": r.get("scheduled_at"),
                "completed_at_s": r.get("completed_at"),
                "latency_s": r.get("latency"),
                "first_token_latency_s": r.get("first_token_latency"),
                "tpot_s": tpot(r),
                "prompt_tokens": r.get("prompt_tokens"),
                "output_tokens": r.get("output_tokens"),
                "num_tokens_received": r.get("num_tokens_received"),
            })
    _write_summary(sibling(stem, "_summary"), {**fields, **_flatten_report(report)})


def write_pooled_latency(stem: Path, latency: LatencyReport, **fields) -> None:
    """Latencies pooled over a point's repeats, to <stem>_pooled_summary.json/.csv."""
    _write_summary(sibling(stem, "_pooled_summary"), {**fields, **_flatten_latency(latency)})


def _write_summary(stem: Path, summary: dict) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    sibling(stem, ".json").write_text(json.dumps(summary, indent=2))
    with sibling(stem, ".csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)


def _flatten_report(report: RunReport) -> dict:
    flat: dict = {
        "duration_s": report.duration_s,
        "wall_clock_s": report.wall_clock_s,
        "requests": report.requests,
        "failures": report.failures,
        "offered_req_s": report.offered_req_s,
        "ttft_growth": report.ttft_growth,
        "keeps_up": report.keeps_up,
        "throughput_req_s": report.throughput_req_s,
        "output_tokens_s": report.output_tokens_s,
        "input_tokens_s": report.input_tokens_s,
        "total_tokens_s": report.total_tokens_s,
        "goodput_req_s": report.goodput_req_s,
        **_flatten_latency(report.latency),
    }
    if report.slo_attainment is not None:
        for name, pct in report.slo_attainment.items():
            flat[f"slo_attainment_{name}_pct"] = pct
    return flat


def _flatten_latency(latency: LatencyReport) -> dict:
    stages = {f"ttft_{shape}": summary for shape, summary in latency.ttft_by_shape.items()}
    stages |= {f"e2e_{shape}": summary for shape, summary in latency.e2e_by_shape.items()}
    stages |= {"tpot": latency.tpot, "e2e_latency": latency.e2e_latency, "itl": latency.itl}
    return {
        f"{stage}_{field_name}": value
        for stage, summary in stages.items() if summary is not None
        for field_name, value in asdict(summary).items()
    }
