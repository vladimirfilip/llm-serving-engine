"""SLO attainment and the largest offered rate an engine sustains."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class Slo:
    ttft_ms: float
    tpot_ms: float

    @classmethod
    def from_config(cls, entry: dict) -> Slo:
        return cls(entry["ttft_ms"], entry["tpot_ms"])


def meets_slo(derived: pd.DataFrame, slo: Slo) -> pd.Series:
    """A request meets the SLO if it succeeded and both latencies are within bounds. A
    single-token request has no TPOT, so only its first-token latency is held to the SLO; a
    longer request whose TPOT could not be measured misses."""
    tpot_ok = (derived.completion_tokens_usage < 2) | (derived.tpot <= slo.tpot_ms / 1000)
    return (derived.status == "ok") & (derived.ttft <= slo.ttft_ms / 1000) & tpot_ok


def max_sustainable_rps(points: pd.DataFrame, target: float) -> float | None:
    """The largest offered rate whose median attainment over repeats is at least `target`,
    with every lower rate meeting it too. `points` has `offered_rps` and `slo_attainment`;
    None when even the lowest rate fails."""
    median_by_rate = points.groupby("offered_rps").slo_attainment.median().sort_index()
    passing = (median_by_rate >= target).to_numpy()
    if not passing[0]:
        return None
    first_fail = np.argmin(passing) if not passing.all() else len(passing)
    return float(median_by_rate.index[first_fail - 1])
