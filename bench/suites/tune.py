"""Stops untuned baselines from making ours look good: every engine's token budget is chosen
by the same procedure, the highest sharegpt output throughput."""

from __future__ import annotations

import json

from ..engines.base import Launch
from ..run import Run
from .common import engine_session, guarded, load_datasets, tuned_path
from .probe import measure_capacity

TIE_TOLERANCE = 0.02


def pick_budget(scores: dict[int, float]) -> int:
    """The best budget, or the smallest among those within 2% of it."""
    best = max(scores.values())
    return min(v for v, s in scores.items() if s >= best * (1 - TIE_TOLERANCE))


def execute(run: Run, engines: list[str]) -> None:
    from ..config import load_engine

    data = load_datasets(run)
    tuned = json.loads(tuned_path(run).read_text()) if tuned_path(run).exists() else {}
    for engine in engines:
        budgets = load_engine(engine, run.cfg.dir).token_budgets
        if budgets == [None]:
            tuned[engine] = {"token_budget": None, "candidates": {}}
            continue
        scores: dict[int, float] = {}
        for budget in budgets:
            cache = run.phase_dir("tune") / f"{engine}_{budget}.json"
            if not cache.exists():
                with guarded(run, "tune", engine), engine_session(
                    run, engine, f"tune{budget}", Launch(token_budget=budget)
                ) as s:
                    cache.write_text(json.dumps(measure_capacity(s, "sharegpt", data)))
            if cache.exists():
                scores[budget] = json.loads(cache.read_text())["capacity_tok_s"]
        if not scores:
            continue
        tuned[engine] = {"token_budget": pick_budget(scores), "candidates": scores}
    tuned_path(run).parent.mkdir(parents=True, exist_ok=True)
    tuned_path(run).write_text(json.dumps(tuned, indent=2))
