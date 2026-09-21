"""Repeats of a point reduced to a median and a min-max range."""

from __future__ import annotations

import pandas as pd


def aggregate_repeats(points: pd.DataFrame, keys: list[str], metrics: list[str]) -> pd.DataFrame:
    """Per key: each metric's median across repeats, with `<metric>_min` and `<metric>_max`.
    Percentiles are taken per repeat before this, never over pooled repeats."""
    grouped = points.groupby(keys)[metrics]
    out = grouped.median()
    for stat in ("min", "max"):
        out = out.join(getattr(grouped, stat)().add_suffix(f"_{stat}"))
    return out.reset_index()
