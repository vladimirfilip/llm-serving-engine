"""What the GPU does between kernels: how busy it is, how long it sits idle, and for ours where
each engine step spends its time. Profiles with Nsight Systems; skipped when it is absent."""

from __future__ import annotations

import shutil
import sqlite3
import subprocess
import threading
import time
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd

from ..client.runner import run_closed_loop
from ..engines.base import Launch
from ..metrics.request_metrics import percentile
from ..run import Run
from .common import SuiteSkipped, engine_session, guarded, load_datasets, tuned_launch
from .single_stream import FixedShapeSource

GAP_LONG_S = 50e-6
STEP_RANGES = ("schedule", "prepare_inputs", "forward", "sample", "postprocess")
DEVICE_TABLES = ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_MEMCPY",
                 "CUPTI_ACTIVITY_KIND_MEMSET")


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """The union of [start, end] intervals across every stream."""
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def busy_and_gaps(intervals: list[tuple[int, int]], window: tuple[int, int]) -> dict:
    """GPU busy fraction of `window` and statistics of the gaps between merged intervals
    (times in seconds). Activity is clipped to the window first, so the fraction cannot exceed 1."""
    lo, hi = window
    merged = merge_intervals([(max(s, lo), min(e, hi)) for s, e in intervals if e > lo and s < hi])
    busy = sum(e - s for s, e in merged)
    gaps = np.array([b_start - a_end for (_, a_end), (b_start, _) in pairwise(merged)]) / 1e9
    span = window[1] - window[0]
    return {"gpu_busy_fraction": busy / span,
            "gap_p50": percentile(gaps, 50), "gap_p99": percentile(gaps, 99),
            "gap_time_fraction_over_50us": float(gaps[gaps > GAP_LONG_S].sum() * 1e9 / span),
            "gaps_s": gaps}


def read_device_intervals(db: sqlite3.Connection) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    present = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    for table in DEVICE_TABLES:
        if table in present:
            out += [(s, e) for s, e in db.execute(f"SELECT start, end FROM {table}")]
    return out


def capture_window(db: sqlite3.Connection, intervals: list[tuple[int, int]],
                   capture_s: float) -> tuple[tuple[int, int], str]:
    """The capture window in ns and where it came from: the start and stop Nsight recorded when
    they bracket the device activity and span about `capture_s`, else `capture_s` from the first
    device event, with the reason the recorded bounds were not used."""
    first = min(s for s, _ in intervals)
    fallback = (first, first + int(capture_s * 1e9))
    reason = "no capture bounds in the export"
    try:
        row = db.execute("SELECT startTime, stopTime FROM ANALYSIS_DETAILS LIMIT 1").fetchone()
    except sqlite3.Error:
        row = None
    if row and row[0] is not None and row[1] is not None:
        span = (row[1] - row[0]) / 1e9
        if row[0] <= first and 0 < span <= 2 * capture_s and span >= 0.5 * capture_s:
            return (row[0], row[1]), "nsight capture bounds"
        reason = f"recorded bounds rejected: span {span:.1f} s for a {capture_s:.1f} s capture"
    return fallback, f"capture_s from the first device event ({reason})"


def read_nvtx(db: sqlite3.Connection) -> pd.DataFrame:
    """NVTX ranges as (name, start, end); the name is inline text or a string-table id,
    depending on the Nsight version. Empty when the capture holds no NVTX events."""
    if not db.execute("SELECT 1 FROM sqlite_master WHERE name='NVTX_EVENTS'").fetchone():
        return pd.DataFrame(columns=["name", "start", "end"])
    columns = {r[1] for r in db.execute("PRAGMA table_info(NVTX_EVENTS)")}
    name = ("COALESCE(text, (SELECT value FROM StringIds WHERE id = textId))"
            if "textId" in columns else "text")
    rows = db.execute(f"SELECT {name}, start, end FROM NVTX_EVENTS WHERE end IS NOT NULL")
    return pd.DataFrame(rows.fetchall(), columns=["name", "start", "end"])


def step_breakdown(nvtx: pd.DataFrame, intervals: list[tuple[int, int]]) -> pd.DataFrame:
    """Per engine step: the period to the next step, the time in each range that starts inside
    it, and the GPU busy time and idle time within the period."""
    steps = nvtx[nvtx.name == "step"].sort_values("start").reset_index(drop=True)
    merged = merge_intervals(intervals)
    rows = []
    for i in range(len(steps) - 1):
        lo, hi = int(steps.start[i]), int(steps.start[i + 1])
        period = hi - lo
        busy = sum(max(0, min(e, hi) - max(s, lo)) for s, e in merged)
        row = {"period_s": period / 1e9, "gpu_busy_s": busy / 1e9,
               "gpu_idle_s": (period - busy) / 1e9}
        for name in STEP_RANGES:
            inside = nvtx[(nvtx.name == name) & (nvtx.start >= lo) & (nvtx.start < hi)]
            row[f"{name}_s"] = float((inside.end - inside.start).sum() / 1e9)
        rows.append(row)
    return pd.DataFrame(rows)


def analyse(db_path: Path, engine: str, batch: int,
            capture_s: float) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    db = sqlite3.connect(db_path)
    intervals = read_device_intervals(db)
    if not intervals:
        return ({"engine": engine, "batch": batch,
                 "window_source": "no device events in the capture"},
                pd.DataFrame(columns=["engine", "batch", "gap_s"]), pd.DataFrame())
    window, source = capture_window(db, intervals, capture_s)
    stats = busy_and_gaps(intervals, window)
    gaps = pd.DataFrame({"engine": engine, "batch": batch, "gap_s": stats.pop("gaps_s")})
    steps = pd.DataFrame()
    if engine == "ours":
        steps = step_breakdown(read_nvtx(db), intervals).assign(batch=batch)
    return {"engine": engine, "batch": batch, "window_source": source} | stats, gaps, steps


def profile(run: Run, engine: str, batch: int, data, folder: Path) -> Path:
    """Profile `batch` closed-loop streams for `capture_s` after `warm_s` of warm-up; returns
    the exported sqlite file."""
    cfg = run.cfg.suite["nsys"]
    nsys = run.cfg.hardware["nsys_path"]
    session = f"bench_{engine}"
    report = folder / f"{engine}_B{batch}"
    launch = tuned_launch(run, engine)
    wrapper = [nsys, "launch", "--trace=cuda,nvtx", "--cuda-graph-trace=node",
               f"--session-new={session}"]
    env = {"ENGINE_NVTX": "1"} if engine == "ours" else {}
    with engine_session(run, engine, f"nsys_B{batch}",
                        Launch(launch.token_budget, launch.args_add, launch.env | env, wrapper),
                        checks=False) as s:
        source = FixedShapeSource(data, 128, 1024, run.cfg.suite["seed"])
        total = cfg["warm_s"] + cfg["capture_s"] + 2
        load = threading.Thread(target=run_closed_loop, daemon=True,
                                args=(s.target(), source, batch, total, s.client_cpus))
        load.start()
        time.sleep(cfg["warm_s"])
        subprocess.run([nsys, "start", f"--session={session}", f"--output={report}",
                        "--force-overwrite=true"], check=True)
        time.sleep(cfg["capture_s"])
        subprocess.run([nsys, "stop", f"--session={session}"], check=True)
        load.join()
    subprocess.run([nsys, "export", "--type=sqlite", f"--output={report}.sqlite",
                    "--force-overwrite=true", f"{report}.nsys-rep"], check=True)
    return Path(f"{report}.sqlite")


def execute(run: Run, engines: list[str]) -> None:
    nsys = run.cfg.hardware["nsys_path"]
    if shutil.which(nsys) is None:
        raise SuiteSkipped(f"{nsys} is not on the path")
    data = load_datasets(run)
    folder = run.phase_dir("nsys")
    summary, gaps, steps = [], [], []
    for engine in engines:
        for batch in run.cfg.suite["nsys"]["batch_sizes"]:
            with guarded(run, "nsys", engine):
                row, gap_frame, step_frame = analyse(
                    profile(run, engine, batch, data, folder), engine, batch,
                    run.cfg.suite["nsys"]["capture_s"])
                summary.append(row)
                gaps.append(gap_frame)
                steps.append(step_frame)
    if not summary:
        return
    tables = run.dir / "tables"
    pd.DataFrame(summary).to_csv(tables / "nsys.csv", index=False)
    pd.concat(gaps).to_parquet(folder / "gaps.parquet")
    pd.concat(steps).to_csv(tables / "nsys_steps.csv", index=False)
