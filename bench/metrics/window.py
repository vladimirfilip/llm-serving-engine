"""One load point reduced to its throughput, latency, goodput and energy metrics over its
measurement window."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .goodput import Slo, meets_slo
from .request_metrics import PERCENTILES, derive, percentile, token_gaps

COMPLETION_MISMATCH_LIMIT = 0.005
PROMPT_MISMATCH_LIMIT = 0.01


def window_slice(derived: pd.DataFrame, window: tuple[float, float]) -> pd.DataFrame:
    """Requests scheduled inside the window."""
    return derived[(derived.t_sched >= window[0]) & (derived.t_sched <= window[1])]


def tokens_in_window(records: pd.DataFrame, window: tuple[float, float]) -> int:
    """Tokens delivered inside the window, including those of requests that arrived earlier
    and of requests that later failed or timed out."""
    lo, hi = window
    return int(sum(((t >= lo) & (t <= hi)).sum() for t in map(np.asarray, records.token_times)))


def point_metrics(records: pd.DataFrame, window: tuple[float, float], slo: Slo, itl_valid: bool,
                  energy_j: float | None = None) -> dict:
    derived = derive(records, itl_valid)
    inside = window_slice(derived, window)
    ok = inside[inside.status == "ok"]
    length = window[1] - window[0]
    tokens = tokens_in_window(records, window)
    completed = derived[(derived.status == "ok") & (derived.t_done >= window[0])
                        & (derived.t_done <= window[1])]
    out = {
        "achieved_rps": len(completed) / length,
        "out_tok_s": tokens / length,
        "slo_attainment": float(meets_slo(inside, slo).mean()) if len(inside) else float("nan"),
        "goodput_rps": int(meets_slo(inside, slo).sum()) / length,
        "error_rate": float((records.status != "ok").mean()),
        "energy_j_per_out_token": energy_j / tokens if energy_j is not None and tokens else None,
    }
    gaps = np.concatenate([token_gaps(t) for t in ok.token_times]) if len(ok) else np.array([])
    for metric, values in (("ttft", ok.ttft), ("tpot", ok.tpot), ("e2e", ok.e2e)):
        for q in PERCENTILES:
            out[f"{metric}_p{q}"] = percentile(values, q)
    for q in PERCENTILES:
        out[f"itl_p{q}"] = percentile(gaps, q) if itl_valid else float("nan")
    return out


def validity(records: pd.DataFrame, lag_limit_ms: float, error_limit: float,
             text_prompts: bool, throttled: bool) -> list[str]:
    """Why a point cannot be trusted; empty when it can. Usage mismatches are measured over
    successful requests only."""
    reasons = []
    lag = percentile(records.t_send - records.t_sched, 99)
    if lag > lag_limit_ms / 1000:
        reasons.append(f"send_lag p99 {lag * 1000:.1f} ms")
    if (records.status != "ok").mean() > error_limit:
        reasons.append("error rate")
    if throttled:
        reasons.append("throttled")
    ok = records[records.status == "ok"]
    if len(ok):
        if (ok.completion_tokens_usage != ok.output_len_req).mean() > COMPLETION_MISMATCH_LIMIT:
            reasons.append("completion_tokens differ from max_tokens")
        skew = ((ok.prompt_tokens_usage - ok.prompt_len).abs() > 1).mean()
        if text_prompts and skew > PROMPT_MISMATCH_LIMIT:
            reasons.append("prompt_tokens differ from the sent prompt")
    return reasons
