"""Writes a run's raw per-request results and aggregate report to disk in both JSON
and CSV, so plots can be regenerated from either without rerunning the benchmark.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

from .report import RunReport, tpot


def write_raw(json_path: Path, csv_path: Path, run: dict) -> None:
    """`run` is `{"target_qps", "duration_s", "results": [...], **extra}` — the same
    shape `plotting.py` reads. Written verbatim to JSON; flattened to one row per
    request for CSV.
    """
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(run, indent=2))

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "success", "error", "latency_s", "first_token_latency_s", "tpot_s",
        "prompt_tokens", "output_tokens", "num_tokens_received",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in run["results"]:
            writer.writerow({
                "success": r.get("success", True),
                "error": r.get("error"),
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
        "throughput_req_s": report.throughput_req_s,
        "output_tokens_s": report.output_tokens_s,
        "input_tokens_s": report.input_tokens_s,
        "total_tokens_s": report.total_tokens_s,
        "goodput_req_s": report.goodput_req_s,
    }
    for stage in ("ttft", "tpot", "e2e_latency", "itl"):
        summary = getattr(report, stage)
        if summary is None:
            continue
        for field_name, value in asdict(summary).items():
            flat[f"{stage}_{field_name}"] = value
    if report.slo_attainment is not None:
        for name, pct in report.slo_attainment.items():
            flat[f"slo_attainment_{name}_pct"] = pct
    return flat
