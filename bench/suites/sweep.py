"""Throughput and latency under open-loop Poisson load, one Parquet file per point."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..client.runner import records_frame, run_open_loop
from ..metrics.goodput import Slo, max_sustainable_rps
from ..metrics.window import point_metrics, validity
from ..run import Run
from ..workloads.arrivals import measurement_window, poisson_schedule, request_count
from ..workloads.workloads import make_requests
from .common import Session, engine_session, guarded, load_datasets
from .probe import ref_capacity_rps, warmup

MAX_CLIENT_PROCS = 8
POINT_ATTEMPTS = 2


def point_paths(
    run: Run, engine: str, workload: str, rate: float, repeat: int
) -> tuple[Path, Path]:
    stem = f"rate{rate:.3f}_rep{repeat}"
    folder = run.dir / "sweep" / engine / workload
    return folder / f"{stem}.parquet", folder / f"{stem}.json"


def point_done(parquet: Path, meta: Path) -> bool:
    """Both files exist and the Parquet holds every request the meta says it should."""
    if not (parquet.exists() and meta.exists()):
        return False
    try:
        rows = len(pd.read_parquet(parquet, columns=["req_id"]))
        return rows == json.loads(meta.read_text())["n"]
    except (OSError, ValueError, KeyError):
        return False


def run_point(session: Session, workload_name: str, rate: float, rate_index: int, repeat: int,
              data) -> None:
    run, cfg = session.run, session.run.cfg
    suite, sweep = cfg.suite, cfg.suite["load_sweep"]
    workload, index = cfg.workloads[workload_name], cfg.workload_index(workload_name)
    per = sweep["per_workload"][workload_name]
    n = request_count(rate, sweep["target_duration_s"], per["min_requests"], per["max_requests"])
    requests = make_requests(workload, index, n, suite["seed"], repeat, data)
    t_sched = poisson_schedule(rate, n, suite["seed"], index, rate_index, repeat)
    window = measurement_window(t_sched, tuple(sweep["window"]))
    slo = Slo.from_config(suite["slo"][workload_name])
    parquet, meta = point_paths(run, session.spec.name, workload_name, rate, repeat)
    parquet.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, POINT_ATTEMPTS + 1):
        warmup(session, workload, index, data, suite["seed"])
        records, t0 = run_open_loop(session.target(), requests, t_sched.tolist(),
                                    session.client_procs, session.client_cpus)
        df = records_frame(records, run_id=run.id, engine=session.spec.name, phase="sweep",
                           workload=workload_name, rate_rps=rate, repeat=repeat)
        throttled = session.monitor.throttled(t0 + window[0], t0 + window[1])
        reasons = validity(df, sweep["client_lag_p99_limit_ms"], sweep["error_rate_limit"],
                           not session.adapter.accepts_token_ids, bool(throttled))
        if not reasons or attempt == POINT_ATTEMPTS:
            break
        if any(r.startswith("send_lag") for r in reasons):
            session.client_procs = min(MAX_CLIENT_PROCS, session.client_procs * 2)
        time.sleep(session.cooldown_s)

    energy = session.monitor.energy_j(t0 + window[0], t0 + window[1])
    metrics = point_metrics(df, window, slo, session.adapter.itl_valid, energy)
    df.drop(columns=["token_ids", "token_logprobs"]).to_parquet(parquet.with_suffix(".tmp"))
    parquet.with_suffix(".tmp").replace(parquet)
    meta.write_text(json.dumps({
        "engine": session.spec.name, "workload": workload_name, "offered_rps": rate,
        "rate_index": rate_index, "repeat": repeat, "n": n, "window": window, "t0": t0,
        "launch": session.adapter.launch_argv, "launch_env": session.adapter.launch_env,
        "token_budget": session.launch.token_budget,
        "monitor": str(session.run.monitor_path(session.spec.name, session.phase)),
        "valid": not reasons, "invalid_reasons": reasons, "throttled": throttled,
        "attempts": attempt, "client_procs": session.client_procs, "metrics": metrics,
    }, indent=2, default=float))
    time.sleep(session.cooldown_s)


def plan(run: Run, engines: list[str]) -> list[tuple[str, int, float, int]]:
    """(workload, rate index, rate, repeat) for every point, shuffled with the suite seed so
    thermal drift does not line up with rate."""
    cfg, sweep = run.cfg, run.cfg.suite["load_sweep"]
    points = []
    for workload in cfg.workloads:
        per = sweep["per_workload"][workload]
        override = (sweep["rate_grid_rps_override"] or {}).get(workload)
        rates = override or [f * ref_capacity_rps(run, workload, engines) for f in per["fractions"]]
        points += [(workload, i, rate, r) for i, rate in enumerate(rates)
                   for r in range(per["repeats"])]
    order = np.random.default_rng(cfg.suite["seed"]).permutation(len(points))
    return [points[i] for i in order]


def execute(run: Run, engines: list[str], workloads: list[str] | None = None) -> None:
    data = load_datasets(run)
    points = [p for p in plan(run, engines) if workloads is None or p[0] in workloads]
    for engine in engines:
        pending = [p for p in points
                   if not point_done(*point_paths(run, engine, p[0], p[2], p[3]))]
        if not pending:
            continue
        with guarded(run, "sweep", engine), engine_session(run, engine, "sweep") as session:
            for workload, rate_index, rate, repeat in pending:
                run_point(session, workload, rate, rate_index, repeat, data)
    write_tables(run, engines)


def load_points(run: Run) -> pd.DataFrame:
    rows = []
    for meta_path in sorted((run.dir / "sweep").glob("*/*/*.json")):
        meta = json.loads(meta_path.read_text())
        rows.append({k: meta[k] for k in ("engine", "workload", "offered_rps", "repeat", "valid",
                                          "throttled", "attempts", "client_procs", "token_budget")}
                    | {"invalid_reasons": "; ".join(meta["invalid_reasons"])} | meta["metrics"])
    return pd.DataFrame(rows)


def write_tables(run: Run, engines: list[str]) -> None:
    points = load_points(run)
    if points.empty:
        return
    points.to_csv(run.dir / "tables" / "sweep_points.csv", index=False)
    target = run.cfg.suite["slo_attainment_target"]
    summary = [{"engine": e, "workload": w, "max_sustainable_rps": max_sustainable_rps(g, target),
                "n_points": len(g), "n_invalid": int((~g.valid).sum())}
               for (e, w), g in points.groupby(["engine", "workload"])]
    pd.DataFrame(summary).to_csv(run.dir / "tables" / "sweep_summary.csv", index=False)

