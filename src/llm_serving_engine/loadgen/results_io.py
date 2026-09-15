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

from .report import RunReport, tpot


def sibling(stem: Path, suffix: str) -> Path:
    return stem.with_name(stem.name + suffix)


def write_raw(json_path: Path, csv_path: Path, run: dict) -> None:
    """`run` is `{"target_qps", "duration_s", "results": [...], **extra}`, the shape the
    plotting functions read. Written verbatim to JSON; flattened to one row per request
    for CSV.
    """
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(run, indent=2))

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "shape", "success", "error", "scheduled_at_s", "completed_at_s", "latency_s",
        "first_token_latency_s", "tpot_s",
        "prompt_tokens", "output_tokens", "num_tokens_received",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
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


def write_summary(json_path: Path, csv_path: Path, report: RunReport, **extra) -> None:
    """Aggregate report for one run, alongside whatever run-identifying `extra` fields
    the caller wants attached (e.g. `target_qps`, `label`)."""
    summary = {**extra, **_flatten_report(report)}

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2))

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)


def _flatten_report(report: RunReport) -> dict:
    flat: dict = {
        "wall_clock_s": report.wall_clock_s,
        "failures": report.failures,
        "latency_growth": report.latency_growth,
        "keeps_up": report.keeps_up,
        "throughput_req_s": report.throughput_req_s,
        "output_tokens_s": report.output_tokens_s,
        "input_tokens_s": report.input_tokens_s,
        "total_tokens_s": report.total_tokens_s,
        "goodput_req_s": report.goodput_req_s,
    }
    stages = {f"ttft_{shape}": summary for shape, summary in report.ttft_by_shape.items()}
    stages |= {"tpot": report.tpot, "e2e_latency": report.e2e_latency, "itl": report.itl}
    for stage, summary in stages.items():
        if summary is None:
            continue
        for field_name, value in asdict(summary).items():
            flat[f"{stage}_{field_name}"] = value
    if report.slo_attainment is not None:
        for name, pct in report.slo_attainment.items():
            flat[f"slo_attainment_{name}_pct"] = pct
    return flat
