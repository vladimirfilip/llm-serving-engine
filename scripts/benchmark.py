"""Benchmark driver: launches `llm-serve` under different `EngineConfig` settings, drives
each with the load generator, and regenerates plots from the raw results written to disk.
The JSON files are the source of truth: rerun this script to update a plot.

Every run starts from the reference config (`_base_env`): a Llama-family model on CUDA with
custom kernels; continuous batching and paged KV are the engine's own defaults. Every load run
sends the load generator's WORKLOAD: mostly short chat turns, plus document requests whose
long prompts and outputs push KV-cache usage toward the pool's capacity.

    python scripts/benchmark.py pareto     --qps 1 2 4
    python scripts/benchmark.py offline
    python scripts/benchmark.py scheduler
    python scripts/benchmark.py allocator
    python scripts/benchmark.py kernels
    python scripts/benchmark.py all

`pareto` sweeps open-loop offered load against the reference config. `offline` measures
closed-loop maximum throughput. Each ablation measures both arms' closed-loop capacity, then
runs both arms open loop at fractions of each capacity, so every arm is compared at loads
it can sustain and at loads it can't, with the latter flagged.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import random
import subprocess
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path

import httpx

from llm_serving_engine.loadgen.cli import add_slo_arguments, slo_from_args
from llm_serving_engine.loadgen.client import (
    WORKLOAD,
    RequestShape,
    load_client,
    request_sender,
)
from llm_serving_engine.loadgen.gpu_monitor import GpuMonitor
from llm_serving_engine.loadgen.kv_monitor import KvCacheMonitor
from llm_serving_engine.loadgen.report import RunReport, build_report
from llm_serving_engine.loadgen.results_io import sibling, write_raw, write_summary
from llm_serving_engine.loadgen.timing import closed_loop_load_gen, open_loop_load_gen
from llm_serving_engine.observability.plotting import (
    plot_ablation_bar,
    plot_ablation_sweep,
    plot_e2e_latency_by_qps,
    plot_goodput_by_qps,
    plot_gpu_memory_by_load,
    plot_kv_utilization_by_load,
    plot_latency_pareto,
    plot_output_throughput_by_qps,
    plot_stage_latency_by_qps,
    plot_throughput_by_qps,
    plot_tpot_by_qps,
    plot_ttft_by_qps,
)
from llm_serving_engine.scheduling.scheduler import MAX_CONCURRENT_SEQUENCES

# Open-loop ablation points, as fractions of each arm's closed-loop capacity.
SWEEP_FRACTIONS = (0.25, 0.5, 0.75, 1.0)


@contextlib.contextmanager
def _running_server(env_overrides: dict[str, str], ready_timeout: float, log_path: Path):
    """Spawns `llm-serve` with `env_overrides` layered on the current environment and its
    output in `log_path`, blocks until `/health` answers, and always tears the process down,
    so a failed run never leaves a stray server holding the port."""
    env = {**os.environ, **env_overrides}
    base_url = f"http://127.0.0.1:{env_overrides['LLM_PORT']}"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        proc = subprocess.Popen(
            [sys.executable, "-m", "llm_serving_engine.server"],
            env=env, stdout=log, stderr=subprocess.STDOUT,
        )
        try:
            _wait_until_healthy(base_url, proc, ready_timeout, log_path)
            yield base_url
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


def _wait_until_healthy(
    base_url: str, proc: subprocess.Popen, timeout: float, log_path: Path
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early with code {proc.returncode}; see {log_path}")
        with contextlib.suppress(httpx.TransportError):
            if httpx.get(f"{base_url}/health", timeout=1.0).status_code == 200:
                return
        time.sleep(0.5)
    raise TimeoutError(f"server did not become healthy within {timeout}s; see {log_path}")


def _logs(out_dir: Path, name: str) -> Path:
    return out_dir / "logs" / f"{name}.log"


def _workload(args: argparse.Namespace) -> list[RequestShape]:
    if args.max_tokens is None:
        return WORKLOAD
    return [replace(shape, max_tokens=args.max_tokens) for shape in WORKLOAD]


def _rng(args: argparse.Namespace, stream: str) -> random.Random:
    return random.Random(f"{args.seed}:{stream}")


def _open_loop(base_url: str, qps: float, args: argparse.Namespace) -> list[dict]:
    async def run() -> list[dict]:
        async with load_client(base_url) as client:
            send = request_sender(client, _rng(args, "shapes"), _workload(args))
            return await open_loop_load_gen(qps, args.duration_s, send, _rng(args, "arrivals"))

    return asyncio.run(run())


def _closed_loop(base_url: str, args: argparse.Namespace) -> list[dict]:
    async def run() -> list[dict]:
        async with load_client(base_url, timeout_s=None) as client:
            send = request_sender(client, _rng(args, "shapes"), _workload(args))
            return await closed_loop_load_gen(args.concurrency, args.duration_s, send)

    return asyncio.run(run())


def _base_env(args: argparse.Namespace) -> dict[str, str]:
    """The reference config every benchmark starts from; an ablation overrides one key."""
    return {
        "LLM_MODEL": args.model,
        "LLM_DEVICE": args.device,
        "LLM_DTYPE": args.dtype,
        "LLM_USE_CUSTOM_KERNELS": "true" if args.use_custom_kernels else "false",
        "LLM_STATIC_BATCH_SIZE": str(args.static_batch_size),
        "LLM_PORT": str(args.port),
    }


def _describe(report: RunReport) -> str:
    e2e_p99 = report.e2e_latency.p99 * 1000 if report.e2e_latency else float("nan")
    ttft_p50 = "".join(
        f" ttft_p50[{shape}]={summary.p50 * 1000:.0f}ms"
        for shape, summary in report.ttft_by_shape.items()
    )
    verdict = "kept up" if report.keeps_up else "did NOT keep up"
    return (
        f"throughput={report.throughput_req_s:.2f} req/s e2e_p99={e2e_p99:.0f}ms{ttft_p50} "
        f"failures={report.failures} latency_growth={report.latency_growth} ({verdict})"
    )


def bench_pareto(args: argparse.Namespace, out_dir: Path) -> None:
    """One engine, offered QPS swept open loop: latency, throughput, goodput, GPU memory
    and KV utilization against offered load."""
    slo = slo_from_args(args)
    runs, reports, gpu_stats, kv_stats = [], [], [], []
    log_path = _logs(out_dir, "pareto")
    with _running_server(_base_env(args), args.ready_timeout, log_path) as base_url:
        for qps in args.qps:
            with GpuMonitor() as gpu, KvCacheMonitor(base_url) as kv:
                results = _open_loop(base_url, qps, args)
            run = {"target_qps": qps, "duration_s": args.duration_s, "results": results}
            report = build_report(results, slo)
            stem = out_dir / "pareto" / f"qps_{qps:g}"
            write_raw(sibling(stem, ".json"), sibling(stem, ".csv"), run)
            write_summary(
                stem.with_name(f"{stem.name}_summary.json"),
                stem.with_name(f"{stem.name}_summary.csv"),
                report, target_qps=qps, duration_s=args.duration_s,
                peak_gpu_memory_mb=gpu.stats.peak_memory_used_mb,
                mean_gpu_utilization_pct=gpu.stats.mean_utilization_pct,
                peak_kv_utilization=kv.stats.peak_utilization,
                mean_kv_utilization=kv.stats.mean_utilization,
                preemptions=kv.stats.preemptions,
            )
            runs.append(run)
            reports.append(report)
            gpu_stats.append(gpu.stats)
            kv_stats.append(kv.stats)
            print(
                f"[pareto] qps={qps:g}: {len(results)} requests, {_describe(report)} "
                f"preemptions={kv.stats.preemptions}"
            )

    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_latency_pareto(runs, str(plot_dir / "pareto.png"))
    plot_stage_latency_by_qps(runs, str(plot_dir / "pareto_stage_latency.png"))
    plot_ttft_by_qps(
        args.qps, [r.ttft_by_shape for r in reports], str(plot_dir / "pareto_ttft.png")
    )
    plot_tpot_by_qps(args.qps, [r.tpot for r in reports], str(plot_dir / "pareto_tpot.png"))
    plot_e2e_latency_by_qps(
        args.qps, [r.e2e_latency for r in reports], str(plot_dir / "pareto_e2e_latency.png")
    )
    plot_throughput_by_qps(
        args.qps, [r.throughput_req_s for r in reports], str(plot_dir / "pareto_throughput.png")
    )
    plot_output_throughput_by_qps(
        args.qps, [r.output_tokens_s for r in reports],
        str(plot_dir / "pareto_output_throughput.png"),
    )
    plot_gpu_memory_by_load(
        args.qps, [g.peak_memory_used_mb for g in gpu_stats],
        str(plot_dir / "pareto_gpu_memory.png"),
    )
    plot_kv_utilization_by_load(
        args.qps, [k.peak_utilization for k in kv_stats],
        str(plot_dir / "pareto_kv_utilization.png"),
    )
    if slo is not None:
        plot_goodput_by_qps(
            args.qps, [r.goodput_req_s for r in reports], str(plot_dir / "pareto_goodput.png")
        )
    print(f"[pareto] wrote plots to {plot_dir}")


def _measure_capacity(
    args: argparse.Namespace, env: dict[str, str], out_stem: Path, log_path: Path
) -> RunReport:
    """Closed-loop maximum throughput. The client has no timeout, so any failure is the
    server's: it is reported with the run, and the server log records why."""
    with _running_server(env, args.ready_timeout, log_path) as base_url:
        with GpuMonitor() as gpu, KvCacheMonitor(base_url) as kv:
            results = _closed_loop(base_url, args)
    report = build_report(results)
    run = {"concurrency": args.concurrency, "duration_s": args.duration_s, "results": results}
    write_raw(sibling(out_stem, ".json"), sibling(out_stem, ".csv"), run)
    write_summary(
        out_stem.with_name(f"{out_stem.name}_summary.json"),
        out_stem.with_name(f"{out_stem.name}_summary.csv"),
        report, concurrency=args.concurrency, duration_s=args.duration_s,
        peak_gpu_memory_mb=gpu.stats.peak_memory_used_mb,
        peak_kv_utilization=kv.stats.peak_utilization,
        preemptions=kv.stats.preemptions,
    )
    if report.failures:
        print(f"{report.failures} closed-loop requests failed; see {log_path}")
    return report


def bench_offline(args: argparse.Namespace, out_dir: Path) -> None:
    """Maximum throughput: `--concurrency` clients kept busy for the whole run."""
    report = _measure_capacity(
        args, _base_env(args), out_dir / "offline" / "offline", _logs(out_dir, "offline")
    )
    print(f"[offline] {_describe(report)}")
    print(
        f"[offline] concurrency={args.concurrency}: "
        f"throughput={report.throughput_req_s:.2f} req/s, "
        f"output={report.output_tokens_s:.1f} tok/s, input={report.input_tokens_s:.1f} tok/s, "
        f"total={report.total_tokens_s:.1f} tok/s"
    )


def _sweep_qps(capacities: list[float]) -> list[float]:
    # An arm that completed nothing has no range to sweep; its capacity bar shows why.
    points = {
        round(f * capacity, 2) for capacity in capacities if capacity > 0 for f in SWEEP_FRACTIONS
    }
    return sorted(points)


def _run_ablation(
    args: argparse.Namespace, out_dir: Path, name: str, env_key: str, arms: list[str]
) -> None:
    """Measures each arm's closed-loop capacity, then drives every arm open loop at the
    union of both arms' sweep points."""
    slo = slo_from_args(args)
    base_env = _base_env(args)
    envs = {arm: {**base_env, env_key: arm} for arm in arms}
    capacity = {
        arm: _measure_capacity(
            args, envs[arm], out_dir / name / arm / "capacity",
            _logs(out_dir, f"{name}_{arm}_capacity"),
        ).throughput_req_s
        for arm in arms
    }
    for arm in arms:
        print(f"[{name}] {arm}: closed-loop capacity {capacity[arm]:.2f} req/s")
    qps_values = _sweep_qps(list(capacity.values()))

    reports: dict[str, list[RunReport]] = {arm: [] for arm in arms}
    for arm in arms:
        log_path = _logs(out_dir, f"{name}_{arm}")
        with _running_server(envs[arm], args.ready_timeout, log_path) as base_url:
            for qps in qps_values:
                results = _open_loop(base_url, qps, args)
                report = build_report(results, slo)
                stem = out_dir / name / arm / f"qps_{qps:g}"
                run = {
                    "target_qps": qps, "duration_s": args.duration_s, env_key: arm,
                    "results": results,
                }
                write_raw(sibling(stem, ".json"), sibling(stem, ".csv"), run)
                write_summary(
                    stem.with_name(f"{stem.name}_summary.json"),
                    stem.with_name(f"{stem.name}_summary.csv"),
                    report, target_qps=qps, duration_s=args.duration_s, **{env_key: arm},
                )
                reports[arm].append(report)
                print(f"[{name}] {arm} qps={qps:g}: {_describe(report)}")

    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    kept_up = {arm: [r.keeps_up for r in reports[arm]] for arm in arms}

    def ms(summary):
        return summary.p99 * 1000 if summary is not None else None

    plot_ablation_bar(
        arms, [capacity[arm] for arm in arms], str(plot_dir / f"{name}_capacity.png"),
        "closed-loop capacity (req/s)",
    )
    plot_ablation_sweep(
        qps_values, {arm: [ms(r.e2e_latency) for r in reports[arm]] for arm in arms}, kept_up,
        str(plot_dir / f"{name}_e2e_p99.png"), "end-to-end latency p99 (ms)",
    )
    for shape in sorted({shape for arm in arms for r in reports[arm] for shape in r.ttft_by_shape}):
        plot_ablation_sweep(
            qps_values,
            {arm: [ms(r.ttft_by_shape.get(shape)) for r in reports[arm]] for arm in arms},
            kept_up,
            str(plot_dir / f"{name}_ttft_p99_{shape}.png"),
            f"{shape} TTFT p99 (ms)",
        )
    plot_ablation_sweep(
        qps_values, {arm: [r.throughput_req_s for r in reports[arm]] for arm in arms}, kept_up,
        str(plot_dir / f"{name}_throughput.png"), "achieved throughput (req/s)",
    )
    if slo is not None:
        plot_ablation_sweep(
            qps_values, {arm: [r.goodput_req_s for r in reports[arm]] for arm in arms}, kept_up,
            str(plot_dir / f"{name}_goodput.png"), "goodput (req/s)",
        )
    print(f"[{name}] wrote plots to {plot_dir}")


def bench_scheduler(args: argparse.Namespace, out_dir: Path) -> None:
    """Continuous vs. static batching."""
    _run_ablation(args, out_dir, "scheduler_ablation", "LLM_SCHEDULER", ["continuous", "static"])


def bench_allocator(args: argparse.Namespace, out_dir: Path) -> None:
    """Paged KV vs. contiguous max-length reservation. Both allocators size off the same
    free GPU memory, so the arms compare at matched memory."""
    _run_ablation(args, out_dir, "allocator_ablation", "LLM_KV_ALLOCATOR", ["paged", "contiguous"])


def bench_kernels(args: argparse.Namespace, out_dir: Path) -> None:
    """Batched Triton kernels (the reference config) vs. the per-sequence HF forward."""
    _run_ablation(args, out_dir, "kernel_ablation", "LLM_USE_CUSTOM_KERNELS", ["true", "false"])


BENCHES = {
    "pareto": bench_pareto,
    "offline": bench_offline,
    "scheduler": bench_scheduler,
    "allocator": bench_allocator,
    "kernels": bench_kernels,
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch llm-serve under different configs and benchmark each."
    )
    parser.add_argument("bench", choices=[*BENCHES, "all"])
    parser.add_argument(
        "--model", default="unsloth/Llama-3.2-3B-Instruct",
        help="model for every server run; matches ModelConfig's own default",
    )
    parser.add_argument("--device", default="cuda", help="LLM_DEVICE for every server run")
    parser.add_argument("--dtype", default="float16", help="LLM_DTYPE for every server run")
    parser.add_argument(
        "--use-custom-kernels", action=argparse.BooleanOptionalAction, default=True,
        help="LLM_USE_CUSTOM_KERNELS for every server run (the kernels ablation overrides this)",
    )
    parser.add_argument("--port", type=int, default=8321)
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--quick", action="store_true",
        help="shrink --duration-s and --qps for fast local iteration, at the cost of a "
        "small per-run sample; p99 on a handful of requests is just max(), so don't use "
        "this for numbers that go in a report",
    )
    parser.add_argument(
        "--duration-s", type=float, default=None,
        help="length of each load run, seconds; long enough at the lowest QPS for about a "
        "hundred completed requests (default: 100, or 30 with --quick)",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="seeds arrivals and request shapes, so every config faces the same workload",
    )
    parser.add_argument(
        "--qps", type=float, nargs="+", default=None,
        help="pareto: QPS points to sweep (default: [1, 2, 4], or [1, 4] with --quick)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=MAX_CONCURRENT_SEQUENCES,
        help="closed-loop clients for offline and ablation capacity runs; the default "
        "matches the engine's running-sequence cap, which saturates it",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=None,
        help="override every request shape's max_tokens (default: each shape's own)",
    )
    parser.add_argument("--static-batch-size", type=int, default=8)
    parser.add_argument(
        "--ready-timeout", type=float, default=120.0, help="seconds to wait for /health"
    )
    add_slo_arguments(parser)
    args = parser.parse_args(argv)
    if args.duration_s is None:
        args.duration_s = 30.0 if args.quick else 100.0
    if args.qps is None:
        args.qps = [1, 4] if args.quick else [1, 2, 4]
    return args


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    failed = []
    for name, bench in BENCHES.items():
        if args.bench not in (name, "all"):
            continue
        try:
            bench(args, args.out_dir)
        except Exception:
            traceback.print_exc()
            failed.append(name)
    if failed:
        sys.exit(f"failed: {', '.join(failed)}; server logs are in {args.out_dir / 'logs'}")


if __name__ == "__main__":
    main()
