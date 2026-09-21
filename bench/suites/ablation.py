"""Ours only: switch its optimizations on one at a time and measure what each buys."""

from __future__ import annotations

import numpy as np
import pandas as pd
import yaml

from ..engines.base import Launch
from ..run import Run
from .common import SuiteSkipped, engine_session, load_datasets, tuned_launch
from .probe import measure_capacity
from .single_stream import decode_tpot, one_at_a_time, prompts

ABLATED_ENGINE = "ours"
PLACEHOLDER = "<flag>"
DECODE_CONTEXT = 512
FULL_REPEATS = (3, 5)  # capacity probe repeats, batch-1 decode repeats
QUICK_REPEATS = (1, 2)


def load_steps(run: Run) -> list[dict]:
    steps = yaml.safe_load((run.cfg.dir / "engines" / "ours_ablation.yaml").read_text())["steps"]
    for step in steps:
        if any(PLACEHOLDER in arg for arg in step.get("args_add", [])):
            raise ValueError(f"ablation step {step['name']!r} still has the {PLACEHOLDER} "
                             f"placeholder: fill in the engine's real flag")
    return steps


def cumulative(steps: list[dict], base: Launch) -> list[tuple[dict, Launch]]:
    """Each step's launch: the tuned budget plus every argument and variable up to that step."""
    args: list[str] = list(base.args_add)
    env: dict[str, str] = dict(base.env)
    out = []
    for step in steps:
        args += step.get("args_add", [])
        env |= {k: str(v) for k, v in step.get("env", {}).items()}
        out.append((step, Launch(base.token_budget, list(args), dict(env))))
    return out


def spread(values: list[float], name: str) -> dict:
    return {name: float(np.median(values)), f"{name}_min": min(values), f"{name}_max": max(values)}


def measure_step(session, data, probe_repeats: int, decode_repeats: int) -> dict:
    capacity = [measure_capacity(session, "sharegpt", data, r)["capacity_tok_s"]
                for r in range(probe_repeats)]
    reqs = prompts(session, data, DECODE_CONTEXT, 128, decode_repeats + 1, 2)
    records = one_at_a_time(session, reqs)[1:]
    tpots = [decode_tpot(r, 16, session.adapter.itl_valid) for r in records if r["status"] == "ok"]
    return spread(capacity, "out_tok_s") | spread(tpots, "tpot_s")


def execute(run: Run, engines: list[str]) -> None:
    if ABLATED_ENGINE not in engines:
        raise SuiteSkipped(f"the ablation runs on {ABLATED_ENGINE} only")
    steps = load_steps(run)
    probe_repeats, decode_repeats = QUICK_REPEATS if run.cfg.quick else FULL_REPEATS
    data = load_datasets(run)
    rows = []
    for step, launch in cumulative(steps, tuned_launch(run, ABLATED_ENGINE)):
        try:
            with engine_session(run, ABLATED_ENGINE, f"ablation_{len(rows)}", launch,
                                checks=False) as s:
                rows.append({"step": step["name"]} | measure_step(s, data, probe_repeats,
                                                                  decode_repeats))
        except RuntimeError as e:
            if not step.get("optional"):
                raise
            rows.append({"step": step["name"], "skipped": str(e).splitlines()[0]})
    pd.DataFrame(rows).to_csv(run.dir / "tables" / "ablation.csv", index=False)
