"""Per-request latencies derived from the client's records. Times are seconds since the
point's shared `t0`; failed and timed-out requests carry NaN latencies."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd

PERCENTILES = (50, 95, 99)


def percentile(values: npt.ArrayLike, q: float) -> float:
    """Linear-interpolated percentile; NaN when there is nothing to rank."""
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    return float(np.percentile(arr, q, method="linear")) if arr.size else float("nan")


def token_gaps(token_times: npt.ArrayLike) -> np.ndarray:
    return np.diff(np.asarray(token_times, dtype=float))


def _tpot(row: pd.Series, itl_valid: bool) -> float:
    """Mean gap after the first token. Without one event per token, tokens are counted from
    the usage event instead of the text events."""
    count = row.n_chunks if itl_valid else row.completion_tokens_usage
    if row.completion_tokens_usage < 2 or count < 2:
        return float("nan")
    return (row.token_times[-1] - row.token_times[0]) / (count - 1)


def derive(records: pd.DataFrame, itl_valid: bool) -> pd.DataFrame:
    """Adds `ttft`, `e2e`, `tpot` and `send_lag`. Latencies are defined only for `ok` rows."""
    df = records.copy()
    ok = df.status == "ok"
    df["send_lag"] = df.t_send - df.t_sched
    df["ttft"] = np.where(ok, df.t_first - df.t_send, np.nan)
    df["e2e"] = np.where(ok, df.t_done - df.t_send, np.nan)
    df["tpot"] = np.nan
    if ok.any():
        df.loc[ok, "tpot"] = df[ok].apply(_tpot, axis=1, itl_valid=itl_valid)
    return df
