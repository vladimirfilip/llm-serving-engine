"""How long an engine takes to serve, and what its first request costs afterwards."""

from __future__ import annotations

import subprocess

import numpy as np
import pandas as pd

from ..client.runner import run_open_loop
from ..run import Run
from ..workloads.workloads import fixed_requests
from .common import engine_session, guarded, load_datasets

PROBE_REQUESTS = 10
PROMPT_TOKENS, OUTPUT_TOKENS = 128, 32
STREAM = 400


def drop_page_cache() -> bool:
    """`sync; echo 3 > /proc/sys/vm/drop_caches`, or False if the process may not."""
    try:
        subprocess.run(["sync"], check=True)
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("3")
    except (OSError, subprocess.CalledProcessError):
        return False
    return True


def first_request_penalty(ttfts: list[float]) -> float:
    """The first request's TTFT less the steady median of the last five."""
    return ttfts[0] - float(np.median(ttfts[5:10]))


def execute(run: Run, engines: list[str]) -> None:
    data = load_datasets(run)
    rows = []
    for engine in engines:
        for launch_index in range(run.cfg.suite["coldstart"]["launches"]):
            dropped = launch_index == 0 and drop_page_cache()
            with guarded(run, "coldstart", engine), engine_session(
                run, engine, f"cold{launch_index}", checks=False
            ) as s:
                reqs = fixed_requests(data.wikitext_train_ids, data.bos_token_id, PROMPT_TOKENS,
                                      OUTPUT_TOKENS, PROBE_REQUESTS, run.cfg.suite["seed"],
                                      STREAM, launch_index)
                ttfts = []
                for req in reqs:
                    (rec,), _ = run_open_loop(s.target(), [req], [0.0], cpus=s.client_cpus)
                    ttfts.append(rec["token_times"][0] - rec["t_send"])
                rows.append({"engine": engine, "launch": launch_index,
                             "cache": "cold_cache" if dropped else "warm_cache",
                             "cache_drop_refused": launch_index == 0 and not dropped,
                             "wait_ready_s": s.ready_s,
                             "first_request_penalty_s": first_request_penalty(ttfts)})
    pd.DataFrame(rows).to_csv(run.dir / "tables" / "coldstart.csv", index=False)
