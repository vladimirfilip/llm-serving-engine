"""Capacity probe: every request submitted at once. The best engine's capacity places the
sweep rates, so every engine faces the same absolute rates."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..client.runner import run_open_loop
from ..run import Run
from ..workloads.workloads import make_requests
from .common import Session, engine_session, guarded, load_datasets

WARMUP_RATE_RPS = 4.0


def warmup(session: Session, workload: dict, workload_index: int, data, seed: int) -> None:
    """Discarded requests at a gentle rate, so first-request costs do not enter a point."""
    n = session.run.cfg.suite["load_sweep"]["warmup_requests"]
    reqs = make_requests(workload, workload_index, n, seed, 1000, data)
    run_open_loop(session.target(), reqs, [i / WARMUP_RATE_RPS for i in range(n)],
                  session.client_procs, session.client_cpus)


def measure_capacity(session: Session, workload_name: str, data, repeat: int = 0) -> dict:
    cfg = session.run.cfg
    workload, index = cfg.workloads[workload_name], cfg.workload_index(workload_name)
    seed = cfg.suite["seed"]
    warmup(session, workload, index, data, seed)
    n = cfg.suite["load_sweep"]["probe_requests"]
    reqs = make_requests(workload, index, n, seed, repeat, data)
    records, _ = run_open_loop(session.target(), reqs, [0.0] * n, session.client_procs,
                               session.client_cpus)
    df = pd.DataFrame(records)
    ok = df[df.status == "ok"]
    span = ok.t_done.max() - df.t_send.min()
    return {"capacity_tok_s": float(ok.completion_tokens_usage.sum() / span),
            "capacity_rps": float(len(ok) / span), "n_ok": len(ok), "n": n}


def capacity_path(run: Run):
    return run.dir / "probe" / "capacity.json"


def load_capacity(run: Run) -> dict:
    return json.loads(capacity_path(run).read_text()) if capacity_path(run).exists() else {}


def ref_capacity_rps(run: Run, workload: str, engines: list[str]) -> float:
    """The largest capacity any engine reached on `workload`."""
    capacity = load_capacity(run)
    missing = [e for e in engines if workload not in capacity.get(e, {})]
    if missing:
        raise RuntimeError(f"no capacity probe for {', '.join(missing)} on {workload}: run "
                           f"`bench run probe --engines ...` first")
    return max(capacity[e][workload]["capacity_rps"] for e in engines)


def execute(run: Run, engines: list[str], repeats: int = 1) -> None:
    data = load_datasets(run)
    capacity = load_capacity(run)
    for engine in engines:
        with guarded(run, "probe", engine), engine_session(run, engine, "probe") as session:
            for workload in run.cfg.workloads:
                results = [measure_capacity(session, workload, data, r) for r in range(repeats)]
                best = results[int(np.argmax([r["capacity_tok_s"] for r in results]))]
                capacity.setdefault(engine, {})[workload] = best | {"repeats": results}
    capacity_path(run).parent.mkdir(exist_ok=True)
    capacity_path(run).write_text(json.dumps(capacity, indent=2))
    rows = [{"engine": e, "workload": w, "capacity_tok_s": c["capacity_tok_s"],
             "capacity_rps": c["capacity_rps"]}
            for e, ws in capacity.items() for w, c in ws.items()]
    pd.DataFrame(rows).to_csv(run.dir / "tables" / "capacity.csv", index=False)
