"""Benchmark driver: launches `llm-serve` under different `EngineConfig` settings, drives
each with the load generator, and regenerates plots from the raw results written to disk.
The JSON files are the source of truth: rerun this script to update a plot.

Every run starts from the reference config (`_base_env`): a Llama-family model on CUDA with
custom kernels; continuous batching and paged KV are the engine's own defaults. Every load run
sends the load generator's WORKLOAD: mostly short chat turns, plus document requests whose
long prompts and outputs push KV-cache usage toward the pool's capacity.

    python scripts/benchmark.py pareto     --qps 1 2 3 4
    python scripts/benchmark.py offline
    python scripts/benchmark.py scheduler
    python scripts/benchmark.py allocator
    python scripts/benchmark.py kernels
    python scripts/benchmark.py all

`pareto` sweeps open-loop offered load against the reference config. `offline` measures
closed-loop maximum throughput. Each ablation measures an arm's closed-loop capacity, then
runs that arm open loop at fractions of its own capacity.

Every load point runs `--repeats` times. Repeat r of every point and every arm draws its
arrivals and request shapes from the same seed, so arms face identical workloads. Rates are
reported per repeat, so their spread shows; latencies pool all repeats' requests. Each
server's output goes to <out-dir>/logs, the only record of why a request failed.
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
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from statistics import median
from typing import Callable

import httpx

from llm_serving_engine.loadgen.cli import add_slo_arguments, slo_from_args
from llm_serving_engine.loadgen.client import (
    WORKLOAD,
    RequestShape,
    load_client,
    request_sender,
    send_request,
)
from llm_serving_engine.loadgen.gpu_monitor import GpuMonitor, GpuStats
from llm_serving_engine.loadgen.kv_monitor import KvCacheMonitor, KvStats
from llm_serving_engine.loadgen.report import PointReport, RunReport, build_point, build_report
from llm_serving_engine.loadgen.results_io import sibling, write_pooled_latency, write_run
from llm_serving_engine.loadgen.timing import closed_loop_load_gen, open_loop_load_gen
from llm_serving_engine.model.sampling import SamplingParams
from llm_serving_engine.observability.plotting import (
    plot_ablation_bar,
    plot_ablation_sweep,
    plot_e2e_latency_by_qps,
    plot_goodput_by_qps,
    plot_gpu_memory_by_load,
    plot_itl_by_qps,
    plot_kv_utilization_by_load,
    plot_latency_pareto,
    plot_output_throughput_by_qps,
    plot_preemptions_by_load,
    plot_stage_latency_by_qps,
    plot_throughput_by_qps,
    plot_tpot_by_qps,
    plot_ttft_by_qps,
    trusted_ms,
)
from llm_serving_engine.scheduling.scheduler import MAX_CONCURRENT_SEQUENCES

# Open-loop ablation points, as fractions of the arm's own closed-loop capacity.
SWEEP_FRACTIONS = (0.25, 0.5, 0.75, 1.0)


@dataclass(slots=True)
class Measured:
    """One load run's raw results and what the monitors saw during it."""

    results: list[dict]
    gpu: GpuStats
    kv: KvStats


@contextlib.contextmanager
def _running_server(env_overrides: dict[str, str], ready_timeout: float, log_path: Path):
    """Spawns `llm-serve` with `env_overrides` layered on the current environment and its
    output in `log_path`, blocks until `/health` answers and a warm-up has run, and always
    tears the process down, so a failed run never leaves a stray server holding the port."""
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
            _warm_up(base_url, log_path)
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


def _warm_up(base_url: str, log_path: Path) -> None:
    """One short request of every shape before anything is measured: the first iteration at
    a new prefill size can JIT-compile Triton kernels, a one-off cost that would otherwise land
    in whichever run goes first."""
    shapes = {shape.name: shape for shape in WORKLOAD}.values()

    async def run() -> list[dict]:
        async with load_client(base_url, timeout_s=None) as client:
            params = SamplingParams(max_tokens=8)
            return await asyncio.gather(*(send_request(client, s.prompt, params) for s in shapes))

    failed = [r["error"] for r in asyncio.run(run()) if not r["success"]]
    if failed:
        raise RuntimeError(f"warm-up requests failed: {failed}; see {log_path}")


def _rng(args: argparse.Namespace, repeat: int, stream: str) -> random.Random:
    return random.Random(f"{args.seed}:{repeat}:{stream}")


def _workload(args: argparse.Namespace) -> list[RequestShape]:
    if args.max_tokens is None:
        return WORKLOAD
    return [replace(shape, max_tokens=args.max_tokens) for shape in WORKLOAD]


def _monitored(base_url: str, load: Callable[[], list[dict]]) -> Measured:
    with GpuMonitor() as gpu, KvCacheMonitor(base_url) as kv:
        results = load()
    return Measured(results, gpu.stats, kv.stats)


def _open_loop(base_url: str, qps: float, args: argparse.Namespace, repeat: int) -> Measured:
    async def run() -> list[dict]:
        async with load_client(base_url) as client:
            send = request_sender(client, _rng(args, repeat, "shapes"), _workload(args))
            return await open_loop_load_gen(
                qps, args.duration_s, send, _rng(args, repeat, "arrivals")
            )

    return _monitored(base_url, lambda: asyncio.run(run()))


def _closed_loop(base_url: str, args: argparse.Namespace, repeat: int) -> Measured:
    async def run() -> list[dict]:
        async with load_client(base_url, timeout_s=None) as client:
            send = request_sender(client, _rng(args, repeat, "shapes"), _workload(args))
            return await closed_loop_load_gen(args.concurrency, args.duration_s, send)

    return _monitored(base_url, lambda: asyncio.run(run()))


def _describe(report: RunReport, kv: KvStats) -> str:
    verdict = {True: "kept up", False: "did NOT keep up", None: "keep-up unknown"}[report.keeps_up]
    offered = "" if report.offered_req_s is None else f"arrivals={report.offered_req_s:.2f}/s "
    ttft_p50 = "".join(
        f" ttft_p50[{shape}]={summary.p50 * 1000:.0f}ms"
        for shape, summary in report.latency.ttft_by_shape.items()
    )
    return (
        f"{offered}throughput={report.throughput_req_s:.2f} req/s "
        f"output={report.output_tokens_s:.0f} tok/s{ttft_p50} failures={report.failures} "
        f"preemptions={kv.preemptions} ttft_growth={report.ttft_growth} ({verdict})"
    )


def _measure_point(
    args: argparse.Namespace,
    stem: Path,
    run_once: Callable[[int], Measured],
    label: str,
    **fields,
) -> tuple[PointReport, list[Measured]]:
    """`--repeats` runs of `run_once`. Each repeat's raw results are written as soon as it
    finishes, so a later crash loses nothing already measured."""
    slo = slo_from_args(args)
    measured: list[Measured] = []
    for repeat in range(args.repeats):
        m = run_once(repeat)
        measured.append(m)
        report = build_report(m.results, args.duration_s, slo)
        run = {
            **fields, "repeat": repeat, "seed": args.seed, "duration_s": args.duration_s,
            "results": m.results,
        }
        write_run(
            sibling(stem, f"_r{repeat}"), run, report, **fields, repeat=repeat,
            peak_gpu_memory_mb=m.gpu.peak_memory_used_mb,
            mean_gpu_utilization_pct=m.gpu.mean_utilization_pct,
            peak_kv_utilization=m.kv.peak_utilization,
            mean_kv_utilization=m.kv.mean_utilization,
            preemptions=m.kv.preemptions,
        )
        print(f"[{label}] repeat {repeat}: {_describe(report, m.kv)}", flush=True)
    point = build_point([m.results for m in measured], args.duration_s, slo)
    write_pooled_latency(stem, point.latency, **fields, repeats=args.repeats)
    return point, measured


def _logs(out_dir: Path, name: str) -> Path:
    return out_dir / "logs" / f"{name}.log"


def bench_pareto(args: argparse.Namespace, out_dir: Path) -> None:
    """One engine, offered QPS swept open loop: latency, throughput, goodput, GPU memory,
    KV utilization and preemptions against offered load."""
    points: list[PointReport] = []
    runs: list[list[Measured]] = []
    with _running_server(_base_env(args), args.ready_timeout, _logs(out_dir, "pareto")) as url:
        for qps in args.qps:
            point, measured = _measure_point(
                args, out_dir / "pareto" / f"qps_{qps:g}", partial(_open_loop, url, qps, args),
                f"pareto qps={qps:g}", target_qps=qps,
            )
            points.append(point)
            runs.append(measured)

    qps = args.qps
    keeps_up = [p.keeps_up for p in points]

    def per_repeat(metric: Callable[[RunReport], float | None]) -> list[list[float | None]]:
        return [[metric(r) for r in p.repeats] for p in points]

    def per_run(metric: Callable[[Measured], float | None]) -> list[list[float | None]]:
        return [[metric(m) for m in measured] for measured in runs]

    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_latency_pareto(
        per_repeat(lambda r: r.throughput_req_s), [p.latency for p in points], keeps_up,
        str(plot_dir / "pareto.png"),
    )
    plot_stage_latency_by_qps(
        [
            {"target_qps": q, "results": [r for m in measured for r in m.results]}
            for q, measured in zip(qps, runs, strict=True)
        ],
        str(plot_dir / "pareto_stage_latency.png"),
    )
    plot_ttft_by_qps(
        qps, [p.latency.ttft_by_shape for p in points], keeps_up, str(plot_dir / "pareto_ttft.png")
    )
    plot_tpot_by_qps(
        qps, [p.latency.tpot for p in points], keeps_up, str(plot_dir / "pareto_tpot.png")
    )
    plot_itl_by_qps(
        qps, [p.latency.itl for p in points], keeps_up, str(plot_dir / "pareto_itl.png")
    )
    plot_e2e_latency_by_qps(
        qps, [p.latency.e2e_latency for p in points], keeps_up,
        str(plot_dir / "pareto_e2e_latency.png"),
    )
    plot_throughput_by_qps(
        qps, per_repeat(lambda r: r.throughput_req_s), per_repeat(lambda r: r.offered_req_s),
        keeps_up, str(plot_dir / "pareto_throughput.png"),
    )
    plot_output_throughput_by_qps(
        qps, per_repeat(lambda r: r.output_tokens_s), keeps_up,
        str(plot_dir / "pareto_output_throughput.png"),
    )
    plot_gpu_memory_by_load(
        qps, per_run(lambda m: m.gpu.peak_memory_used_mb), keeps_up,
        str(plot_dir / "pareto_gpu_memory.png"),
    )
    plot_kv_utilization_by_load(
        qps, per_run(lambda m: m.kv.peak_utilization), per_run(lambda m: m.kv.mean_utilization),
        keeps_up, str(plot_dir / "pareto_kv_utilization.png"),
    )
    plot_preemptions_by_load(
        qps, per_run(lambda m: m.kv.preemptions), keeps_up,
        str(plot_dir / "pareto_preemptions.png"),
    )
    if slo_from_args(args) is not None:
        plot_goodput_by_qps(
            qps, per_repeat(lambda r: r.goodput_req_s), keeps_up,
            str(plot_dir / "pareto_goodput.png"),
        )
    print(f"[pareto] wrote plots to {plot_dir}")


def bench_offline(args: argparse.Namespace, out_dir: Path) -> None:
    """Maximum throughput: `--concurrency` clients kept busy for the whole run."""
    with _running_server(_base_env(args), args.ready_timeout, _logs(out_dir, "offline")) as url:
        point, _ = _measure_point(
            args, out_dir / "offline" / "offline", partial(_closed_loop, url, args), "offline",
            concurrency=args.concurrency,
        )
    print(f"[offline] concurrency={args.concurrency}: {_spread(point, 'throughput_req_s')} req/s, "
          f"{_spread(point, 'output_tokens_s')} output tok/s, "
          f"{_spread(point, 'total_tokens_s')} total tok/s")


def _spread(point: PointReport, rate: str) -> str:
    values = [getattr(r, rate) for r in point.repeats]
    return f"median {median(values):.2f} (min {min(values):.2f}, max {max(values):.2f})"


def _run_ablation(
    args: argparse.Namespace, out_dir: Path, name: str, env_key: str, arms: list[str]
) -> None:
    """For each arm on one server: closed-loop capacity, then open loop at SWEEP_FRACTIONS
    of that arm's median capacity. Arms of very different capacity then each get a sweep
    spanning their own range, instead of one arm idling and the other drowning."""
    capacities: dict[str, list[float]] = {}
    qps_by_arm: dict[str, list[float]] = {}
    points_by_arm: dict[str, list[PointReport]] = {}
    for arm in arms:
        env = {**_base_env(args), env_key: arm}
        with _running_server(env, args.ready_timeout, _logs(out_dir, f"{name}_{arm}")) as url:
            capacity, _ = _measure_point(
                args, out_dir / name / arm / "capacity", partial(_closed_loop, url, args),
                f"{name} {arm} capacity", concurrency=args.concurrency, **{env_key: arm},
            )
            capacities[arm] = [r.throughput_req_s for r in capacity.repeats]
            print(f"[{name}] {arm}: closed-loop capacity {_spread(capacity, 'throughput_req_s')}")
            arm_capacity = median(capacities[arm])
            # An arm that completed nothing has no range to sweep; its capacity bar shows why.
            qps_by_arm[arm] = (
                [round(f * arm_capacity, 3) for f in SWEEP_FRACTIONS] if arm_capacity > 0 else []
            )
            points_by_arm[arm] = [
                _measure_point(
                    args, out_dir / name / arm / f"qps_{qps:g}",
                    partial(_open_loop, url, qps, args), f"{name} {arm} qps={qps:g}",
                    target_qps=qps, **{env_key: arm},
                )[0]
                for qps in qps_by_arm[arm]
            ]

    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    keeps_up = {arm: [p.keeps_up for p in points_by_arm[arm]] for arm in arms}

    def sweep(metric: Callable[[PointReport], float | None], filename: str, ylabel: str) -> None:
        plot_ablation_sweep(
            qps_by_arm, {arm: [metric(p) for p in points_by_arm[arm]] for arm in arms},
            keeps_up, str(plot_dir / f"{name}_{filename}.png"), ylabel,
        )

    def median_rate(rate: Callable[[RunReport], float | None]):
        def metric(point: PointReport) -> float | None:
            values = [v for v in (rate(r) for r in point.repeats) if v is not None]
            return median(values) if values else None

        return metric

    plot_ablation_bar(
        arms, [capacities[arm] for arm in arms], str(plot_dir / f"{name}_capacity.png"),
        "closed-loop capacity (req/s)",
    )
    sweep(lambda p: trusted_ms(p.latency.e2e_latency, "p99"), "e2e_p99", "end-to-end p99 (ms)")
    sweep(lambda p: trusted_ms(p.latency.itl, "p99"), "itl_p99", "inter-token latency p99 (ms)")
    shapes = sorted(
        {shape for arm in arms for p in points_by_arm[arm] for shape in p.latency.ttft_by_shape}
    )
    for shape in shapes:
        sweep(
            lambda p, shape=shape: trusted_ms(p.latency.ttft_by_shape.get(shape), "p50"),
            f"ttft_p50_{shape}", f"{shape} TTFT p50 (ms)",
        )
    sweep(median_rate(lambda r: r.throughput_req_s), "throughput", "achieved throughput (req/s)")
    if slo_from_args(args) is not None:
        sweep(median_rate(lambda r: r.goodput_req_s), "goodput", "goodput (req/s)")
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
        help="one 30 s repeat per point over fewer QPS points, for fast local iteration; "
        "too few requests for tail percentiles, so don't use it for numbers that go in a report",
    )
    parser.add_argument(
        "--duration-s", type=float, default=None,
        help="length of each load run, seconds (default: 100, or 30 with --quick)",
    )
    parser.add_argument(
        "--repeats", type=int, default=None,
        help="runs per load point; rates report their spread across repeats and latencies "
        "pool them (default: 3, or 1 with --quick)",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="seeds arrivals and request shapes; repeat r uses the same draws at every point",
    )
    parser.add_argument(
        "--qps", type=float, nargs="+", default=None,
        help="pareto: QPS points to sweep (default: 1 2 2.5 3 3.5 4, or 1 3 with --quick)",
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
    if args.repeats is None:
        args.repeats = 1 if args.quick else 3
    if args.qps is None:
        args.qps = [1, 3] if args.quick else [1, 2, 2.5, 3, 3.5, 4]
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
