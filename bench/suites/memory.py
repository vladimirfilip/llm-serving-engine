"""Memory: where it goes, how much load fits, how well the KV pool is used, and what happens
when it runs out."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

from ..client.runner import run_open_loop
from ..run import Run
from ..workloads.arrivals import poisson_schedule, request_count
from ..workloads.workloads import Request, fixed_requests, make_requests
from .common import Session, engine_session, guarded, load_datasets
from .probe import load_capacity, ref_capacity_rps
from .session_tools import Relauncher, StatsPoller

GRID_STATUSES = ("ok", "preempted", "failed", "not run")
GRID_POLL_HZ = 20
KV_POLL_HZ = 10
STREAM = 200  # rng stream index of the memory suite's own workloads


def classify_exhaustion(records: list[dict], elapsed_s: float, server_exited: bool,
                        hang_s: float) -> str:
    """What overload did: `server_crashed` if the process is gone; `completed_all` if every
    request finished; `hang` if requests are left and no stream produced a token for `hang_s`
    before the run ended; otherwise `partial_errors`."""
    if server_exited:
        return "server_crashed"
    if all(r["status"] == "ok" for r in records):
        return "completed_all"
    last_token = max((r["token_times"][-1] for r in records if r["token_times"]), default=0.0)
    return "hang" if elapsed_s - last_token >= hang_s else "partial_errors"


def preemptions(samples: list[dict]) -> int | None:
    """Preemptions counted over a run of stats samples, None where the engine has no counter."""
    counts = [s["preemptions_total"] for s in samples if s.get("preemptions_total") is not None]
    return int(counts[-1] - counts[0]) if counts else None


def peak_running(samples: list[dict]) -> int | None:
    values = [s["running"] for s in samples if s.get("running") is not None]
    return int(max(values)) if values else None


def cell_requests(data, length: int, batch: int, output: int, seed: int) -> list[Request]:
    return fixed_requests(data.wikitext_train_ids, data.bos_token_id, length - output, output,
                          batch, seed, STREAM + length % 89, batch)


def run_cell(session: Session, data, length: int, batch: int, cfg: dict) -> dict:
    """`batch` requests of `length` tokens at once: whether they all finish, whether the
    engine had to preempt, and the most it ran together."""
    output = cfg["cell_output_tokens"]
    reqs = cell_requests(data, length, batch, output, session.run.cfg.suite["seed"])
    target = session.target(timeout_s=cfg["cell_timeout_s"])
    with StatsPoller(session.adapter, GRID_POLL_HZ) as poller:
        records, _ = run_open_loop(target, reqs, [0.0] * batch, session.client_procs,
                                   session.client_cpus)
    crashed = session.adapter.proc.exited()
    rose = preemptions(poller.samples)
    if crashed or any(r["status"] != "ok" for r in records):
        status = "failed"
    else:
        status = "preempted" if rose else "ok"
    return {"length": length, "batch": batch, "status": status,
            "peak_running": peak_running(poller.samples), "preemptions": rose}


def capacity_grid(run: Run, engine: str, data, folder: Path) -> list[dict]:
    """Lengths ascending, batches ascending; a row stops at its first failure. Finished cells
    are kept on disk, so a rerun resumes."""
    cfg, max_len = run.cfg.suite["memory"], run.cfg.model["max_model_len"]
    path = folder / "grid.jsonl"
    done = {(c["length"], c["batch"]): c for c in map(json.loads, path.read_text().splitlines())} \
        if path.exists() else {}
    with Relauncher(run, engine, "memory_grid") as engine_up:
        for length in cfg["grid_lengths"]:
            failed = length > max_len
            for batch in cfg["grid_batches"]:
                if (length, batch) in done:
                    failed = failed or done[(length, batch)]["status"] == "failed"
                    continue
                if failed:
                    cell = {"length": length, "batch": batch, "status": "not run",
                            "peak_running": None, "preemptions": None,
                            "reason": "exceeds max_model_len" if length > max_len else "row failed"}
                else:
                    cell = run_cell(engine_up.session(), data, length, batch, cfg)
                    failed = cell["status"] == "failed"
                done[(length, batch)] = cell
                with path.open("a") as f:
                    f.write(json.dumps(cell) + "\n")
    return list(done.values())


def sharegpt_load(session: Session, data, rate: float, duration_s: float, repeat: int) -> tuple:
    """Open-loop ShareGPT arrivals at `rate` for `duration_s`; the `perf_counter` span it took."""
    cfg, seed = session.run.cfg, session.run.cfg.suite["seed"]
    n = request_count(rate, duration_s, 1, 10**6)
    reqs = make_requests(cfg.workloads["sharegpt"], cfg.workload_index("sharegpt"), n, seed,
                         repeat, data)
    t_sched = poisson_schedule(rate, n, seed, STREAM, 0, repeat)
    start = time.perf_counter()
    run_open_loop(session.target(), reqs, t_sched.tolist(), session.client_procs,
                  session.client_cpus)
    return start, time.perf_counter()


def kv_series(samples: list[dict], t0: float) -> pd.DataFrame:
    rows = []
    for s in samples:
        kv = s["kv"]
        if kv.get("blocks_total"):
            util = kv["blocks_used"] / kv["blocks_total"]
            used = kv["blocks_used"] * kv["block_size"]
            frag = 1 - kv["tokens_used"] / used if used else 0.0
        else:
            util, frag = kv.get("usage"), None
        rows.append({"t_s": s["t"] - t0, "kv_util": util, "internal_frag": frag})
    return pd.DataFrame(rows)


def device_used(session: Session) -> int | None:
    used = session.monitor.frame().get("mem_used_bytes")
    return int(used.dropna().iloc[-1]) if used is not None and used.notna().any() else None


def measure_utilization(
    session: Session, data, capacity: dict, ref_rps: float, folder: Path
) -> dict:
    """The breakdown and KV occupancy of an engine under sustained ShareGPT load, and the
    peak device memory it reached."""
    cfg = session.run.cfg.suite["memory"]
    time.sleep(1.5)  # one GPU-monitor status sample after startup
    stats = session.adapter.stats()
    out = {"idle_device_bytes": device_used(session),
           "idle_breakdown": stats and stats.get("memory_bytes")}
    if session.spec.name == "ours" and "ours" in capacity:
        half = 0.5 * capacity["ours"]["sharegpt"]["capacity_rps"]
        sharegpt_load(session, data, half, cfg["kv_util_duration_s"], 1)
        loaded = session.adapter.stats()
        out["loaded_breakdown"] = loaded and loaded.get("memory_bytes")
    with StatsPoller(session.adapter, KV_POLL_HZ) as poller:
        start, end = sharegpt_load(session, data, cfg["kv_util_rate_fraction"] * ref_rps,
                                   cfg["kv_util_duration_s"], 2)
    kv_series(poller.samples, start).to_parquet(folder / "kv_util.parquet")
    out["peak_device_bytes"] = session.monitor.peak_memory_bytes(start, end)
    return out


def exhaustion(session: Session, cfg: dict, max_model_len: int) -> dict:
    """Overload past the KV pool: the outcome, whether the engine recovers, and its preemptions."""
    output = cfg["exhaustion_output_tokens"]
    reqs = [Request(i, [1, *range(2 + i, 2 + i + max_model_len - output - 2)], output)
            for i in range(cfg["exhaustion_batch"])]
    before = session.adapter.stats()
    started = time.perf_counter()
    with StatsPoller(session.adapter, KV_POLL_HZ) as poller:
        records, _ = run_open_loop(session.target(timeout_s=cfg["cell_timeout_s"]), reqs,
                                   [0.0] * len(reqs), session.client_procs, session.client_cpus)
    elapsed = time.perf_counter() - started
    outcome = classify_exhaustion(records, elapsed, session.adapter.proc.exited(), cfg["hang_s"])
    kv = (before or {}).get("kv") or {}
    capacity_tokens = (kv.get("blocks_total") or 0) * (kv.get("block_size") or 0)
    return {"outcome": outcome, "recovered": recovered(session, cfg["recover_s"]),
            "oversubscription": (cfg["exhaustion_batch"] * max_model_len / capacity_tokens
                                 if capacity_tokens else None),
            "preemptions": preemptions(poller.samples)}


def recovered(session: Session, recover_s: float) -> bool:
    """Whether `/health` is up and a 16-token request completes within `recover_s` of the
    overload."""
    deadline = time.perf_counter() + recover_s
    while time.perf_counter() < deadline:
        if session.adapter.proc.exited():
            return False
        try:
            with urllib.request.urlopen(session.adapter.base_url()
                                        + session.spec.health_path, timeout=5):
                pass
            body = session.adapter.complete([1, 2, 3], 16)
            if body["usage"]["completion_tokens"] == 16:
                return True
        except (urllib.error.URLError, OSError, KeyError, ValueError):
            pass
        time.sleep(1.0)
    return False


def sweep_peak_memory(run: Run, engine: str) -> int | None:
    """Peak device memory across the ShareGPT sweep point at the engine's highest passing rate."""
    summary = run.dir / "tables" / "sweep_summary.csv"
    if not summary.exists():
        return None
    table = pd.read_csv(summary)
    row = table[(table.engine == engine) & (table.workload == "sharegpt")]
    if row.empty or pd.isna(row.max_sustainable_rps.iloc[0]):
        return None
    for meta_path in (run.dir / "sweep" / engine / "sharegpt").glob("*.json"):
        meta = json.loads(meta_path.read_text())
        if abs(meta["offered_rps"] - row.max_sustainable_rps.iloc[0]) < 1e-9:
            frame = pd.read_parquet(meta["monitor"])
            lo, hi = meta["t0"] + meta["window"][0], meta["t0"] + meta["window"][1]
            used = frame[(frame.t >= lo) & (frame.t <= hi)].mem_used_bytes.dropna()
            return int(used.max()) if len(used) else None
    return None


def execute(run: Run, engines: list[str]) -> None:
    data = load_datasets(run)
    cfg = run.cfg.suite["memory"]
    capacity = load_capacity(run)
    ref_rps = ref_capacity_rps(run, "sharegpt", engines)
    breakdown, device, grid, exhausted = [], [], [], []
    for engine in engines:
        folder = run.phase_dir("memory") / engine
        folder.mkdir(exist_ok=True)
        util = None
        with guarded(run, "memory", engine), engine_session(run, engine, "memory_util") as s:
            util = measure_utilization(s, data, capacity, ref_rps, folder)
        if util is None:
            continue
        for phase in ("idle", "loaded"):
            for component, nbytes in (util.get(f"{phase}_breakdown") or {}).items():
                breakdown.append({"engine": engine, "phase": phase, "component": component,
                                  "bytes": nbytes})
        sweep_peak = sweep_peak_memory(run, engine)
        device.append({"engine": engine, "idle_bytes": util["idle_device_bytes"],
                       "peak_bytes": sweep_peak or util["peak_device_bytes"],
                       "peak_source": "sweep" if sweep_peak else "kv_util run"})
        with guarded(run, "memory", engine):
            grid += [{"engine": engine, **c} for c in capacity_grid(run, engine, data, folder)]
        with guarded(run, "memory", engine), Relauncher(run, engine, "memory_exhaustion") as up:
            exhausted.append({"engine": engine,
                              **exhaustion(up.session(), cfg, run.cfg.model["max_model_len"])})
    tables = run.dir / "tables"
    for name, rows in (("memory_breakdown", breakdown), ("memory_device", device),
                       ("memory_grid", grid), ("memory_exhaustion", exhausted)):
        pd.DataFrame(rows).to_csv(tables / f"{name}.csv", index=False)
    kv = [pd.read_parquet(p).assign(engine=p.parent.name)
          for p in (run.dir / "memory").glob("*/kv_util.parquet")]
    if kv:
        pd.concat(kv).to_csv(tables / "kv_util.csv", index=False)
