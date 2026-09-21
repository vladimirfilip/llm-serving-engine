"""Scheduler behaviour under stress: what a long prefill does to streams already decoding, and
who gets starved when the engine is offered more than it can serve."""

from __future__ import annotations

import asyncio
import time
from itertools import pairwise

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..client.runner import _use_uvloop, run_open_loop
from ..client.stream import Target, new_session, send
from ..engines.base import Launch
from ..metrics.request_metrics import percentile
from ..run import Run
from ..workloads.arrivals import poisson_schedule, request_count
from ..workloads.workloads import Request, make_requests
from .common import Session, engine_session, guarded, load_datasets, tuned_launch
from .probe import ref_capacity_rps

STREAM = 300
TPOT_WINDOW_S = 2.0


def during_prefill(gap_start: float, gap_end: float, injected: list[tuple[float, float]]) -> bool:
    """Whether a token gap overlaps the [send, first token] interval of any injected request."""
    return any(gap_start < end and gap_end > start for start, end in injected)


def label_gaps(background: list[dict], injected: list[tuple[float, float]]) -> pd.DataFrame:
    """Every inter-token gap of the background streams, labelled by whether an injected prefill
    was in progress while it elapsed."""
    rows = []
    for stream, rec in enumerate(background):
        times = rec["token_times"]
        for a, b in pairwise(times):
            rows.append({"stream": stream, "t": b, "itl": b - a,
                         "label": "during_prefill" if during_prefill(a, b, injected)
                         else "baseline"})
    return pd.DataFrame(rows, columns=["stream", "t", "itl", "label"])


def windowed_tpot(background: list[dict], window_s: float,
                  injected: list[tuple[float, float]]) -> pd.DataFrame:
    """Where gaps are not one per token, per-stream TPOT over fixed windows stands in for ITL,
    labelled by the same overlap rule as gaps."""
    rows = []
    for stream, rec in enumerate(background):
        times = np.asarray(rec["token_times"])
        for start in np.arange(0.0, times.max() if len(times) else 0.0, window_s):
            inside = times[(times >= start) & (times < start + window_s)]
            if len(inside) > 1:
                overlapping = during_prefill(start, start + window_s, injected)
                rows.append({"stream": stream, "t": start + window_s,
                             "itl": (inside[-1] - inside[0]) / (len(inside) - 1),
                             "label": "during_prefill" if overlapping else "baseline"})
    return pd.DataFrame(rows, columns=["stream", "t", "itl", "label"])


async def interference_run(target: Target, source, cfg: dict, t0: float) -> tuple[list, list]:
    """Background closed-loop streams for the whole run, and one long prefill injected on a
    fixed cadence in the middle."""
    total = cfg["warm_s"] + cfg["inject_for_s"] + cfg["tail_s"]
    end = t0 + total
    background: list[dict] = []

    async def stream_worker(session, index: int) -> None:
        while time.perf_counter() < end:
            background.append(await send(session, target, source(index), 0.0, t0, open_loop=False))

    async def injector(session) -> list[dict]:
        injected, tasks = [], []
        offset = 0.0
        while offset < cfg["inject_for_s"]:
            req = Request(len(tasks), source.inject_prompt(), cfg["inject_output"])
            tasks.append(asyncio.create_task(
                send(session, target, req, cfg["warm_s"] + offset, t0)))
            offset += cfg["inject_every_s"]
        injected = await asyncio.gather(*tasks)
        return list(injected)

    async with new_session() as session:
        workers = [asyncio.create_task(stream_worker(session, i))
                   for i in range(cfg["background_streams"])]
        injected = await injector(session)
        await asyncio.gather(*workers)
    return background, injected


class InterferenceSource:
    """Fixed-shape requests for the background streams and the injected long prompt."""

    def __init__(self, data, cfg: dict, seed: int):
        self.data, self.cfg = data, cfg
        self.rng = np.random.default_rng([seed, STREAM])
        self.issued = 0

    def _ids(self, length: int) -> list[int]:
        start = int(self.rng.integers(0, len(self.data.wikitext_train_ids) - length))
        body = self.data.wikitext_train_ids[start : start + length - 1].tolist()
        return [self.data.bos_token_id, *body]

    def __call__(self, _worker: int) -> Request:
        self.issued += 1
        return Request(self.issued, self._ids(self.cfg["background_prompt"]),
                       self.cfg["background_output"])

    def inject_prompt(self) -> list[int]:
        return self._ids(self.cfg["inject_prompt"])


def summarize_interference(gaps: pd.DataFrame, injected: list[dict]) -> list[dict]:
    rows = []
    for label, group in gaps.groupby("label", dropna=False):
        itl = group.itl.to_numpy()
        rows.append({"label": label, "n": len(itl), "itl_p50": percentile(itl, 50),
                     "itl_p99": percentile(itl, 99), "itl_max": float(itl.max())})
    ttft = [r["token_times"][0] - r["t_send"] for r in injected if r["status"] == "ok"]
    extra = {"inject_ttft_p50": percentile(ttft, 50), "inject_ttft_max": max(ttft, default=None)}
    return [row | extra for row in rows]


def run_interference(session: Session, data, variant: str, folder) -> list[dict]:
    cfg = session.run.cfg.suite["scheduler"]["interference"]
    _use_uvloop()
    source = InterferenceSource(data, cfg, session.run.cfg.suite["seed"])
    t0 = time.perf_counter()
    background, injected = asyncio.run(interference_run(session.target(), source, cfg, t0))
    windows = [(r["t_send"], r["t_first"]) for r in injected if r["status"] == "ok"]
    gaps = (label_gaps(background, windows) if session.adapter.itl_valid
            else windowed_tpot(background, TPOT_WINDOW_S, windows))
    gaps.assign(variant=variant).to_parquet(folder / f"interference_{variant}.parquet")
    return summarize_interference(gaps, injected)


def overload_metrics(records: list[dict], starvation_ttft_s: float) -> dict:
    """A request that never got a first token counts as starved, the worst case there is."""
    df = pd.DataFrame(records)
    ok = df[df.status == "ok"]
    ttft = (ok.t_first - ok.t_send).to_numpy()
    waited = df.t_first - df.t_send
    starved = (waited > starvation_ttft_s) | (waited.isna() & (df.status != "ok"))
    rho = spearmanr(ok.prompt_len, ttft).statistic if len(ok) > 2 else float("nan")
    p50, p99 = percentile(ttft, 50), percentile(ttft, 99)
    return {"n": len(df), "ttft_p50": p50, "ttft_p99": p99,
            "ttft_max": float(ttft.max()) if len(ttft) else float("nan"),
            "ttft_p99_over_p50": p99 / p50 if p50 else float("nan"),
            "timeout_rate": float((df.status == "timeout").mean()),
            "starved_fraction": float(starved.mean()),
            "spearman_prompt_len_ttft": float(rho),
            "send_lag_p99": percentile(df.t_send - df.t_sched, 99)}


def run_overload(session: Session, data, ref_rps: float, folder) -> dict:
    cfg = session.run.cfg
    over = cfg.suite["scheduler"]["overload"]
    rate = over["rate_fraction"] * ref_rps
    seed = cfg.suite["seed"]
    n = request_count(rate, over["duration_s"], 1, 10**6)
    reqs = make_requests(cfg.workloads["sharegpt"], cfg.workload_index("sharegpt"), n, seed, 3,
                         data)
    t_sched = poisson_schedule(rate, n, seed, STREAM, 1, 0)
    records, _ = run_open_loop(session.target(timeout_s=over["timeout_s"]), reqs,
                               t_sched.tolist(), session.client_procs, session.client_cpus)
    pd.DataFrame(records).drop(columns=["token_times", "token_ids", "token_logprobs"]).to_parquet(
        folder / "overload.parquet")
    return overload_metrics(records, over["starvation_ttft_s"])


def variants_of(run: Run, engine: str) -> dict[str, Launch]:
    """The launches to compare: the engine's tuned default and each variant its YAML defines."""
    from ..config import load_engine

    base = tuned_launch(run, engine)
    out = {"default": base}
    for name, v in load_engine(engine, run.cfg.dir).variants.items():
        out[name] = Launch(base.token_budget, list(v.get("args_add", [])), dict(v.get("env", {})))
    return out


def execute(run: Run, engines: list[str]) -> None:
    data = load_datasets(run)
    interference, overload = [], []
    for engine in engines:
        folder = run.phase_dir("scheduler") / engine
        folder.mkdir(exist_ok=True)
        for variant, launch in variants_of(run, engine).items():
            with guarded(run, "scheduler", engine), engine_session(
                run, engine, f"scheduler_{variant}", launch
            ) as s:
                interference += [{"engine": engine, "variant": variant} | row
                                 for row in run_interference(s, data, variant, folder)]
        with guarded(run, "scheduler", engine), engine_session(run, engine, "overload") as s:
            ref = ref_capacity_rps(run, "sharegpt", engines)
            overload.append({"engine": engine} | run_overload(s, data, ref, folder))
    tables = run.dir / "tables"
    pd.DataFrame(interference).to_csv(tables / "scheduler_interference.csv", index=False)
    pd.DataFrame(overload).to_csv(tables / "scheduler_overload.csv", index=False)
