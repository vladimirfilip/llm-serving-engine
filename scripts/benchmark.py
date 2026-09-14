"""Benchmark driver: launches `llm-serve` under different `EngineConfig` knobs,
drives each instance with the real open-loop load generator, and regenerates plots
from the raw results written to disk. The JSON files are the source of truth: rerun this
script to update a plot.

Every run starts from the same reference config (`_base_env`): a Llama-family model,
fused forward, continuous batching, paged KV. Continuous and paged are the engine's own
defaults, so only `--model`/`--device`/`--dtype`/`--use-custom-kernels`
need setting here. Each two-arm ablation overrides exactly one of those axes for its
second arm; the rest stay at the reference default so the comparison is isolated.

    python scripts/benchmark.py pareto     --qps 1 2 4
    python scripts/benchmark.py offline    --offline-qps 50
    python scripts/benchmark.py scheduler  --ablation-qps 2
    python scripts/benchmark.py allocator  --ablation-qps 2
    python scripts/benchmark.py kernels    --ablation-qps 2
    python scripts/benchmark.py all

`pareto` additionally reports TTFT/TPOT/goodput/GPU/KV-cache against offered rate, and
`offline` approximates an always-full-queue max-throughput run (see `bench_offline`).
Every run's raw per-request results and aggregate report are written as JSON and CSV.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

from llm_serving_engine.loadgen.client import LoadGenConfig, send_request
from llm_serving_engine.loadgen.gpu_monitor import GpuMonitor
from llm_serving_engine.loadgen.kv_monitor import KvUtilizationMonitor
from llm_serving_engine.loadgen.report import SLOThresholds, build_report
from llm_serving_engine.loadgen.results_io import write_raw, write_summary
from llm_serving_engine.loadgen.timing import open_loop_load_gen
from llm_serving_engine.model.sampling import SamplingParams
from llm_serving_engine.observability.metrics import summarize
from llm_serving_engine.observability.plotting import (
    plot_ablation_bar,
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


@contextlib.contextmanager
def _running_server(env_overrides: dict[str, str], ready_timeout: float):
    """Spawns `llm-serve` as a subprocess with `env_overrides` layered on the current
    environment, blocks until `/health` answers, and always tears the process down, so
    a failed ablation run never leaves a stray server holding the port."""
    env = {**os.environ, **env_overrides}
    base_url = f"http://127.0.0.1:{env_overrides['LLM_PORT']}"
    proc = subprocess.Popen(
        [sys.executable, "-m", "llm_serving_engine.server"],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        _wait_until_healthy(base_url, proc, ready_timeout)
        yield base_url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def _wait_until_healthy(base_url: str, proc: subprocess.Popen, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server process exited early with code {proc.returncode}")
        with contextlib.suppress(httpx.TransportError):
            if httpx.get(f"{base_url}/health", timeout=1.0).status_code == 200:
                return
        time.sleep(0.5)
    raise TimeoutError(f"server did not become healthy within {timeout}s")


async def _run_load(
    base_url: str, target_qps: float, duration_s: float, max_tokens: int | None
) -> list[dict]:
    sampling_params = SamplingParams(max_tokens=max_tokens) if max_tokens is not None else None
    config = LoadGenConfig(
        target_qps=target_qps, duration_s=duration_s, base_url=base_url, sampling_params=sampling_params
    )

    async def send_fn() -> dict:
        return await send_request(config.base_url, config.sample_prompt(), config.sampling_params)

    return await open_loop_load_gen(config.target_qps, config.duration_s, send_fn)


def _write_run(path: Path, target_qps: float, duration_s: float, results: list[dict], **extra) -> None:
    run = {"target_qps": target_qps, "duration_s": duration_s, "results": results, **extra}
    write_raw(path.with_suffix(".json"), path.with_suffix(".csv"), run)


def _slo_from_args(args: argparse.Namespace) -> SLOThresholds | None:
    if args.slo_ttft_ms is None and args.slo_tpot_ms is None and args.slo_e2e_ms is None:
        return None
    return SLOThresholds(ttft_ms=args.slo_ttft_ms, tpot_ms=args.slo_tpot_ms, e2e_ms=args.slo_e2e_ms)


def _achieved_throughput(results: list[dict], duration_s: float) -> float:
    return sum(1 for r in results if r.get("success", True)) / duration_s


def _p99(results: list[dict]) -> float:
    latencies = [r["latency"] for r in results if r.get("success", True)]
    return summarize(latencies).p99 if latencies else float("nan")


def _base_env(args: argparse.Namespace) -> dict[str, str]:
    """The reference config every benchmark starts from: a Llama-family model, fused
    forward, on CUDA. Each ablation overrides exactly one of these keys for its second
    arm; everything else stays at this default so the comparison isolates one axis."""
    return {
        "LLM_MODEL": args.model,
        "LLM_DEVICE": args.device,
        "LLM_DTYPE": args.dtype,
        "LLM_USE_CUSTOM_KERNELS": "true" if args.use_custom_kernels else "false",
        "LLM_PORT": str(args.port),
    }


def bench_pareto(args: argparse.Namespace, out_dir: Path) -> None:
    """One engine, offered QPS swept: latency/throughput/goodput/GPU/KV all reported
    against the offered rate, plus the throughput-latency Pareto and stage-latency
    breakdown."""
    slo = _slo_from_args(args)
    runs, reports, gpu_stats, kv_stats = [], [], [], []
    with _running_server(_base_env(args), args.ready_timeout) as base_url:
        for qps in args.qps:
            with GpuMonitor() as gpu, KvUtilizationMonitor(base_url) as kv:
                results = asyncio.run(_run_load(base_url, qps, args.duration_s, args.max_tokens))
            run = {"target_qps": qps, "duration_s": args.duration_s, "results": results}
            report = build_report(results, args.duration_s, slo)
            _write_run(out_dir / "pareto" / f"qps_{qps:g}", qps, args.duration_s, results)
            write_summary(
                out_dir / "pareto" / f"qps_{qps:g}_summary.json",
                out_dir / "pareto" / f"qps_{qps:g}_summary.csv",
                report, target_qps=qps, duration_s=args.duration_s,
                peak_gpu_memory_mb=gpu.stats.peak_memory_used_mb,
                mean_gpu_utilization_pct=gpu.stats.mean_utilization_pct,
                peak_kv_utilization=kv.stats.peak_utilization,
                mean_kv_utilization=kv.stats.mean_utilization,
            )
            runs.append(run)
            reports.append(report)
            gpu_stats.append(gpu.stats)
            kv_stats.append(kv.stats)
            print(f"[pareto] qps={qps:g}: {len(results)} requests, "
                  f"throughput={report.throughput_req_s:.2f} req/s")

    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_latency_pareto(runs, str(plot_dir / "pareto.png"))
    plot_stage_latency_by_qps(runs, str(plot_dir / "pareto_stage_latency.png"))
    plot_ttft_by_qps(args.qps, [r.ttft for r in reports], str(plot_dir / "pareto_ttft.png"))
    plot_tpot_by_qps(args.qps, [r.tpot for r in reports], str(plot_dir / "pareto_tpot.png"))
    plot_e2e_latency_by_qps(
        args.qps, [r.e2e_latency for r in reports], str(plot_dir / "pareto_e2e_latency.png")
    )
    plot_throughput_by_qps(
        args.qps, [r.throughput_req_s for r in reports], str(plot_dir / "pareto_throughput.png")
    )
    plot_output_throughput_by_qps(
        args.qps, [r.output_tokens_s for r in reports], str(plot_dir / "pareto_output_throughput.png")
    )
    plot_gpu_memory_by_load(
        args.qps, [g.peak_memory_used_mb for g in gpu_stats], str(plot_dir / "pareto_gpu_memory.png")
    )
    plot_kv_utilization_by_load(
        args.qps, [k.peak_utilization for k in kv_stats], str(plot_dir / "pareto_kv_utilization.png")
    )
    if slo is not None:
        plot_goodput_by_qps(
            args.qps, [r.goodput_req_s for r in reports], str(plot_dir / "pareto_goodput.png")
        )
    print(f"[pareto] wrote plots to {plot_dir}")


def bench_offline(args: argparse.Namespace, out_dir: Path) -> None:
    """Offline/max-throughput mode: an always-full request queue, approximated by
    running the existing open-loop generator at an offered rate well above what the
    server can sustain, so arrivals queue continuously. Only throughput is reported: a
    saturated open-loop run is no steady-state per-request latency measurement."""
    with _running_server(_base_env(args), args.ready_timeout) as base_url:
        with GpuMonitor() as gpu, KvUtilizationMonitor(base_url) as kv:
            results = asyncio.run(
                _run_load(base_url, args.offline_qps, args.duration_s, args.max_tokens)
            )
    report = build_report(results, args.duration_s)
    _write_run(out_dir / "offline" / "offline", args.offline_qps, args.duration_s, results)
    write_summary(
        out_dir / "offline" / "offline_summary.json", out_dir / "offline" / "offline_summary.csv",
        report, offered_qps=args.offline_qps, duration_s=args.duration_s,
        peak_gpu_memory_mb=gpu.stats.peak_memory_used_mb,
        peak_kv_utilization=kv.stats.peak_utilization,
    )
    print(
        f"[offline] {len(results)} requests: throughput={report.throughput_req_s:.2f} req/s, "
        f"output={report.output_tokens_s:.1f} tok/s, input={report.input_tokens_s:.1f} tok/s, "
        f"total={report.total_tokens_s:.1f} tok/s"
    )


def _run_ablation(
    args: argparse.Namespace, out_dir: Path, ablation_name: str, env_key: str, labels: list[str]
) -> None:
    """Shared driver for the three two-arm ablations: same offered load against each
    labeled config, starting from the same reference `_base_env`, so `env_key` is the
    only thing that differs between arms."""
    base_env = _base_env(args)
    throughputs, p99s = [], []
    for label in labels:
        env = {**base_env, env_key: label, "LLM_STATIC_BATCH_SIZE": str(args.static_batch_size)}
        with _running_server(env, args.ready_timeout) as base_url:
            results = asyncio.run(_run_load(base_url, args.ablation_qps, args.duration_s, args.max_tokens))
        _write_run(
            out_dir / ablation_name / f"{label}.json",
            args.ablation_qps, args.duration_s, results, **{env_key: label},
        )
        throughput, p99 = _achieved_throughput(results, args.duration_s), _p99(results) * 1000
        throughputs.append(throughput)
        p99s.append(p99)
        print(f"[{ablation_name}] {label}: throughput={throughput:.2f} req/s p99={p99:.1f}ms")

    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_ablation_bar(
        labels, throughputs, str(plot_dir / f"{ablation_name}_throughput.png"),
        "achieved throughput (req/s)",
    )
    plot_ablation_bar(labels, p99s, str(plot_dir / f"{ablation_name}_p99.png"), "p99 latency (ms)")
    print(f"[{ablation_name}] wrote {plot_dir}/{ablation_name}_throughput.png and _p99.png")


def bench_scheduler(args: argparse.Namespace, out_dir: Path) -> None:
    """Static vs. continuous batching, same offered load: the achieved-throughput gap
    between the two is the number worth reporting."""
    _run_ablation(args, out_dir, "scheduler_ablation", "LLM_SCHEDULER", ["continuous", "static"])


def bench_allocator(args: argparse.Namespace, out_dir: Path) -> None:
    """Contiguous max-length reservation vs. paged KV cache: both allocators are sized
    off the same KVCacheConfig, so holding model/device/gpu-memory-utilization fixed
    across arms compares the two at matched memory."""
    _run_ablation(args, out_dir, "allocator_ablation", "LLM_KV_ALLOCATOR", ["paged", "contiguous"])


def bench_kernels(args: argparse.Namespace, out_dir: Path) -> None:
    """Fused batched decode path (the reference default) vs. the default per-sequence
    HF decode loop: same offered load, same Llama-family model."""
    _run_ablation(args, out_dir, "kernel_ablation", "LLM_USE_CUSTOM_KERNELS", ["true", "false"])


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch llm-serve under different configs and benchmark each with the "
        "open-loop load generator."
    )
    parser.add_argument(
        "bench", choices=["pareto", "offline", "scheduler", "allocator", "kernels", "all"]
    )
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
        help="run length per config, seconds; long enough at the lowest configured QPS "
        "for a few hundred completed requests, since p99 on a handful of samples is just "
        "max() (default: 300, or 60 with --quick)",
    )
    parser.add_argument(
        "--qps", type=float, nargs="+", default=None,
        help="pareto: QPS points to sweep (default: [1, 2, 4], or [1, 4] with --quick)",
    )
    parser.add_argument(
        "--ablation-qps", type=float, default=2.0,
        help="scheduler/allocator/kernels ablations: offered QPS",
    )
    parser.add_argument(
        "--offline-qps", type=float, default=50.0,
        help="offline: offered QPS, well above server capacity so the queue stays full",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=32, help="cap generated tokens per request"
    )
    parser.add_argument("--static-batch-size", type=int, default=8)
    parser.add_argument("--ready-timeout", type=float, default=120.0, help="seconds to wait for /health")
    parser.add_argument("--slo-ttft-ms", type=float, default=None, help="goodput: TTFT SLO, ms")
    parser.add_argument("--slo-tpot-ms", type=float, default=None, help="goodput: TPOT SLO, ms")
    parser.add_argument("--slo-e2e-ms", type=float, default=None, help="goodput: end-to-end SLO, ms")
    args = parser.parse_args(argv)
    if args.duration_s is None:
        args.duration_s = 60.0 if args.quick else 300.0
    if args.qps is None:
        args.qps = [1, 4] if args.quick else [1, 2, 4]
    return args


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.bench in ("pareto", "all"):
        bench_pareto(args, args.out_dir)
    if args.bench in ("offline", "all"):
        bench_offline(args, args.out_dir)
    if args.bench in ("scheduler", "all"):
        bench_scheduler(args, args.out_dir)
    if args.bench in ("allocator", "all"):
        bench_allocator(args, args.out_dir)
    if args.bench in ("kernels", "all"):
        bench_kernels(args, args.out_dir)


if __name__ == "__main__":
    main()
