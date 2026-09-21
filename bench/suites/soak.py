"""A long open-loop run at a steady fraction of capacity: does memory grow, does throughput
drift, does the tail get worse, does anything fail."""

from __future__ import annotations

import asyncio
import time

import numpy as np
import pandas as pd

from ..client.runner import _use_uvloop
from ..client.stream import Target, new_session, send
from ..run import Run
from ..workloads.workloads import make_requests
from .common import SuiteSkipped, engine_session, guarded, load_datasets
from .probe import ref_capacity_rps

STREAM = 500
CHUNK_S = 60.0
THROUGHPUT_WINDOW_S = 60.0
LATENCY_WINDOW_S = 300.0
# Minutes 10-60 of a four-hour run are the reference and 60-240 the evaluation; fractions of
# the run keep the comparison the same at any duration.
REFERENCE_FRACTION = (10 / 240, 60 / 240)
EVALUATION_START_FRACTION = 60 / 240


async def soak_loop(target: Target, requests_for_chunk, rate: float, duration_s: float, seed: int,
                    t0: float) -> list[dict]:
    """Poisson arrivals generated a chunk at a time, so hours of load hold only a chunk of
    schedule and requests in memory at once."""
    rng = np.random.default_rng([seed, STREAM])
    tasks: list[asyncio.Task] = []
    async with new_session() as session:
        chunk_start = 0.0
        while chunk_start < duration_s:
            span = min(CHUNK_S, duration_s - chunk_start)
            gaps = rng.exponential(1 / rate, size=max(1, int(rate * span * 2)))
            offsets = chunk_start + np.cumsum(gaps)
            offsets = offsets[offsets < chunk_start + span]
            reqs = requests_for_chunk(len(offsets), int(chunk_start / CHUNK_S))
            tasks += [asyncio.create_task(send(session, target, r, float(t), t0))
                      for r, t in zip(reqs, offsets, strict=True)]
            chunk_start += span
            await asyncio.sleep(max(0.0, t0 + chunk_start - CHUNK_S - time.perf_counter()))
        return list(await asyncio.gather(*tasks))


def windowed(records: pd.DataFrame, window_s: float, duration_s: float) -> pd.DataFrame:
    """Per fixed window: output tokens per second, and TPOT and TTFT p50/p99 of the requests
    that completed in it."""
    rows = []
    ok = records[records.status == "ok"]
    tokens = np.concatenate([np.asarray(t) for t in ok.token_times]) if len(ok) else np.array([])
    for start in np.arange(0.0, duration_s, window_s):
        done = ok[(ok.t_done >= start) & (ok.t_done < start + window_s)]
        tpot = ((done.token_times.map(lambda t: t[-1]) - done.t_first)
                / (done.completion_tokens_usage - 1).clip(lower=1)).to_numpy()
        ttft = (done.t_first - done.t_send).to_numpy()
        rows.append({"t_s": start, "out_tok_s": float(((tokens >= start)
                                                       & (tokens < start + window_s)).sum()
                                                      / window_s),
                     "tpot_p50": float(np.percentile(tpot, 50)) if len(tpot) else np.nan,
                     "tpot_p99": float(np.percentile(tpot, 99)) if len(tpot) else np.nan,
                     "ttft_p50": float(np.percentile(ttft, 50)) if len(ttft) else np.nan,
                     "ttft_p99": float(np.percentile(ttft, 99)) if len(ttft) else np.nan})
    return pd.DataFrame(rows)


def soak_verdict(tokens: pd.DataFrame, latency: pd.DataFrame, memory: pd.DataFrame,
                 errors: int, duration_s: float, limits: dict) -> dict:
    """The pass criteria: the evaluation part of the run against its reference part."""
    ref_lo, ref_hi = (f * duration_s for f in REFERENCE_FRACTION)
    eval_lo = EVALUATION_START_FRACTION * duration_s

    def part(frame: pd.DataFrame, column: str, lo: float, hi: float) -> np.ndarray:
        inside = frame[(frame.t_s >= lo) & (frame.t_s < hi)][column].dropna()
        return inside.to_numpy()

    mem_ref = np.median(part(memory, "mem_used_bytes", ref_lo, ref_hi))
    mem_growth = (part(memory, "mem_used_bytes", eval_lo, duration_s).max() - mem_ref) / mem_ref
    tok_ref = np.median(part(tokens, "out_tok_s", ref_lo, ref_hi))
    tok_drift = abs(np.median(part(tokens, "out_tok_s", eval_lo, duration_s)) - tok_ref) / tok_ref
    p99_ref = np.median(part(latency, "tpot_p99", ref_lo, ref_hi))
    p99_growth = (np.median(part(latency, "tpot_p99", eval_lo, duration_s)) - p99_ref) / p99_ref
    checks = {
        "gpu_mem_growth": (mem_growth, limits["gpu_mem_growth_max_frac"]),
        "throughput_drift": (tok_drift, limits["throughput_drift_max_frac"]),
        "p99_tpot_growth": (p99_growth, limits["p99_tpot_drift_max_frac"]),
    }
    out = {name: {"value": float(v), "limit": lim, "passed": bool(v <= lim)}
           for name, (v, lim) in checks.items()}
    out["errors"] = {"value": errors, "limit": 0, "passed": errors == 0}
    out["passed"] = all(c["passed"] for c in out.values())
    return out


def execute(run: Run, engines: list[str]) -> None:
    cfg = run.cfg.suite["soak"]
    wanted = [e for e in engines if e in cfg["engines"]]
    if not wanted:
        raise SuiteSkipped(f"soak runs on {', '.join(cfg['engines'])} only")
    data = load_datasets(run)
    duration_s = cfg["duration_h"] * 3600
    ref = ref_capacity_rps(run, "sharegpt", engines)
    workload = run.cfg.workloads["sharegpt"]
    index = run.cfg.workload_index("sharegpt")
    for engine in wanted:
        folder = run.phase_dir("soak") / engine
        folder.mkdir(exist_ok=True)
        with guarded(run, "soak", engine), engine_session(run, engine, "soak") as s:
            def requests_for_chunk(n: int, chunk: int):
                return make_requests(workload, index, n, run.cfg.suite["seed"], chunk, data)

            _use_uvloop()
            t0 = time.perf_counter()
            records = asyncio.run(soak_loop(s.target(), requests_for_chunk,
                                            cfg["rate_fraction"] * ref, duration_s,
                                            run.cfg.suite["seed"], t0))
            monitor = s.monitor.frame()
        df = pd.DataFrame(records)
        tokens = windowed(df, THROUGHPUT_WINDOW_S, duration_s)
        latency = windowed(df, LATENCY_WINDOW_S, duration_s)
        memory = monitor.dropna(subset=["mem_used_bytes"]).copy()
        memory["t_s"] = memory.t - t0
        memory = memory.iloc[:: max(1, int(cfg["sample_s"]))]
        tokens.to_csv(folder / "throughput.csv", index=False)
        latency.to_csv(folder / "latency.csv", index=False)
        memory[["t_s", "mem_used_bytes", "server_rss_bytes"]].to_csv(folder / "memory.csv",
                                                                     index=False)
        verdict = soak_verdict(tokens, latency, memory, int((df.status != "ok").sum()),
                               duration_s, run.cfg.suite["soak_pass"])
        pd.Series(verdict).to_json(folder / "verdict.json")
        pd.DataFrame([{"engine": engine, "passed": verdict["passed"]}
                      | {k: v["value"] for k, v in verdict.items() if k != "passed"}]
                     ).to_csv(run.dir / "tables" / "soak.csv", index=False)
