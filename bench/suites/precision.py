"""Ours at each precision the model config lists: what quality it costs and what throughput it
buys."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..engines.base import Launch
from ..run import Run
from .common import SuiteSkipped, engine_session, load_datasets, tuned_launch
from .correctness import measure_perplexity, run_gsm8k
from .probe import measure_capacity

PRECISIONED_ENGINE = "ours"


def execute(run: Run, engines: list[str]) -> None:
    precisions = run.cfg.model["precisions"]
    if len(precisions) < 2:
        raise SuiteSkipped("model.yaml lists a single precision")
    if PRECISIONED_ENGINE not in engines:
        raise SuiteSkipped(f"precision variants run on {PRECISIONED_ENGINE} only")
    data = load_datasets(run)
    base = tuned_launch(run, PRECISIONED_ENGINE)
    rows = []
    for precision in precisions:
        launch = Launch(base.token_budget, base.args_add, base.env | {"LLM_DTYPE": precision})
        with engine_session(run, PRECISIONED_ENGINE, f"precision_{precision}", launch) as s:
            folder = run.phase_dir("precision") / precision
            rows.append({"precision": precision,
                         "capacity_tok_s": measure_capacity(s, "sharegpt", data)["capacity_tok_s"],
                         "ppl": measure_perplexity(s)} | (run_gsm8k(s, folder) or {}))
    table = pd.DataFrame(rows)
    table["ppl_delta_vs_first"] = table.ppl - table.ppl.iloc[0] if table.ppl.notna().all() \
        else np.nan
    table.to_csv(run.dir / "tables" / "precision.csv", index=False)
