"""CLI entrypoint for the load generator (`llm-loadgen` console script).

Owns argument parsing and writing raw per-request results to disk; see plotting.py
for reading those files back into plots.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

from ..model.sampling import SamplingParams
from .client import LoadGenConfig, send_request
from .report import SLOThresholds, build_report
from .results_io import write_raw, write_summary
from .timing import open_loop_load_gen


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open-loop load generator for the serving engine")
    parser.add_argument("--target-qps", type=float, required=True, help="offered load, requests/s")
    parser.add_argument(
        "--duration-s", type=float, required=True,
        help="run length, seconds; pick this so target-qps * duration-s is in the hundreds, "
        "since p99 on a handful of samples is just max()",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="engine base URL")
    parser.add_argument("--max-tokens", type=int, default=None, help="override SamplingParams")
    parser.add_argument("--temperature", type=float, default=None, help="override SamplingParams")
    parser.add_argument("--top-p", type=float, default=None, help="override SamplingParams")
    parser.add_argument(
        "--out", type=Path, default=None, help="output path stem (default: results_<ts>); "
        "writes <stem>.json/.csv raw and <stem>_summary.json/.csv aggregate"
    )
    parser.add_argument("--slo-ttft-ms", type=float, default=None, help="goodput: TTFT SLO, ms")
    parser.add_argument("--slo-tpot-ms", type=float, default=None, help="goodput: TPOT SLO, ms")
    parser.add_argument("--slo-e2e-ms", type=float, default=None, help="goodput: end-to-end SLO, ms")
    return parser.parse_args(argv)


def _sampling_params_from_args(args: argparse.Namespace) -> SamplingParams | None:
    overrides = {"max_tokens": args.max_tokens, "temperature": args.temperature, "top_p": args.top_p}
    given = {k: v for k, v in overrides.items() if v is not None}
    return SamplingParams(**given) if given else None


def _slo_from_args(args: argparse.Namespace) -> SLOThresholds | None:
    if args.slo_ttft_ms is None and args.slo_tpot_ms is None and args.slo_e2e_ms is None:
        return None
    return SLOThresholds(ttft_ms=args.slo_ttft_ms, tpot_ms=args.slo_tpot_ms, e2e_ms=args.slo_e2e_ms)


async def _run(config: LoadGenConfig) -> list[dict]:
    async def send_fn() -> dict:
        return await send_request(config.base_url, config.sample_prompt(), config.sampling_params)

    return await open_loop_load_gen(config.target_qps, config.duration_s, send_fn)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    config = LoadGenConfig(
        target_qps=args.target_qps,
        duration_s=args.duration_s,
        base_url=args.base_url,
        sampling_params=_sampling_params_from_args(args),
    )
    results = asyncio.run(_run(config))

    # One run per file, tagged with its offered load: exactly the shape
    # plotting.plot_latency_pareto consumes, so a sweep is regenerable from the raw files.
    run = {"target_qps": config.target_qps, "duration_s": config.duration_s, "results": results}
    stem = (args.out or Path(f"results_{int(time.time())}")).with_suffix("")
    write_raw(stem.with_suffix(".json"), stem.with_suffix(".csv"), run)

    report = build_report(results, config.duration_s, _slo_from_args(args))
    summary_stem = stem.with_name(f"{stem.name}_summary")
    write_summary(
        summary_stem.with_suffix(".json"), summary_stem.with_suffix(".csv"), report,
        target_qps=config.target_qps, duration_s=config.duration_s,
    )
    print(f"wrote {len(results)} results to {stem}.json/.csv, summary to {summary_stem}.json/.csv")


if __name__ == "__main__":
    main()
