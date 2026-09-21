"""Shared look and plumbing of every plot: colours, titles, footers, watermarks, captions."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from ..metrics.aggregate import aggregate_repeats

COLORS = {"ours": "#111111", "vllm": "#1f77b4", "sglang": "#2ca02c", "trtllm": "#d62728",
          "mock": "#9467bd", "torch": "#8c564b", "flash_attn": "#e377c2", "flashinfer": "#17becf"}
DRAW_ORDER = ["mock", "vllm", "sglang", "trtllm", "ours"]  # ours last, so it is on top
PANEL = (7, 4.5)
DPI = 200


def color(engine: str) -> str:
    return COLORS.get(engine, "#7f7f7f")


def ordered(engines) -> list[str]:
    names = set(engines)
    known = [e for e in DRAW_ORDER if e in names]
    return known + sorted(names - set(known))


@dataclass(slots=True)
class PlotContext:
    """Everything a plot needs: the run's tables and environment, and what to stamp on it."""

    run_dir: Path
    out_dir: Path
    env: dict
    suite: dict
    baseline_versions: dict = field(default_factory=dict)
    correctness_failed: bool = False
    quick: bool = False
    _cache: dict = field(default_factory=dict)

    def table(self, name: str) -> pd.DataFrame | None:
        """`tables/<name>.csv`, or None when that suite did not produce it."""
        if name not in self._cache:
            path = self.run_dir / "tables" / f"{name}.csv"
            try:
                frame = pd.read_csv(path) if path.exists() else None
            except pd.errors.EmptyDataError:  # a suite wrote an empty frame: no columns at all
                frame = None
            self._cache[name] = frame if frame is not None and len(frame) else None
        return self._cache[name]

    def json(self, relative: str) -> dict | None:
        path = self.run_dir / relative
        return json.loads(path.read_text()) if path.exists() else None

    def sweep(self, workload: str, metrics: list[str]) -> pd.DataFrame | None:
        """Sweep points of one workload with each metric's median over repeats and its min
        and max, plus whether any repeat at that rate was invalid."""
        points = self.table("sweep_points")
        if points is None:
            return None
        points = points[points.workload == workload]
        if points.empty:
            return None
        flags = points.groupby(["engine", "offered_rps"]).valid.all().rename("valid")
        agg = aggregate_repeats(points, ["engine", "offered_rps"], metrics)
        return agg.merge(flags.reset_index(), on=["engine", "offered_rps"]).sort_values(
            ["engine", "offered_rps"])

    @property
    def title_suffix(self) -> str:
        gpu = self.env.get("gpu", {}).get("name", "no GPU")
        model = Path(self.env["model"]["path"]).name
        return f"{model} - {gpu} - {self.env['model']['dtype']}"


def to_ms(frame: pd.DataFrame, *prefixes: str) -> pd.DataFrame:
    """A copy with every column starting with one of `prefixes` converted from seconds to ms."""
    out = frame.copy()
    for column in out.columns:
        if column.startswith(prefixes):
            out[column] = out[column] * 1000
    return out


def figure(nrows: int = 1, ncols: int = 1, **kw):
    fig, axes = plt.subplots(nrows, ncols, figsize=(PANEL[0] * ncols, PANEL[1] * nrows),
                             squeeze=False, **kw)
    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)
    return fig, axes


def line(ax, engine: str, x, y, **kw):
    """One engine's line: ours thicker, in the engine's fixed colour."""
    kw.setdefault("marker", "o")
    kw.setdefault("markersize", 4)
    kw.setdefault("label", engine)
    return ax.plot(x, y, color=color(engine), lw=3 if engine == "ours" else 1.6,
                   zorder=5 if engine == "ours" else 3, **kw)


def points(ax, engine: str, frame: pd.DataFrame, x: str, y: str, yerr: bool = True) -> bool:
    """A line with a marker per row, hollow where the point was invalid or throttled, and
    min-max error bars where repeats exist. True if any marker was hollow."""
    frame = frame.sort_values(x)
    line(ax, engine, frame[x], frame[y], marker=None)
    if yerr and f"{y}_min" in frame:
        spread = (frame[y] - frame[f"{y}_min"]).abs() + (frame[f"{y}_max"] - frame[y]).abs()
        if (spread > 0).any():
            ax.errorbar(frame[x], frame[y], yerr=[frame[y] - frame[f"{y}_min"],
                                                  frame[f"{y}_max"] - frame[y]],
                        fmt="none", ecolor=color(engine), alpha=0.6, capsize=2)
    valid = frame.get("valid", pd.Series(True, index=frame.index)).astype(bool)
    ax.plot(frame[x][valid], frame[y][valid], "o", color=color(engine), ms=5, zorder=6)
    ax.plot(frame[x][~valid], frame[y][~valid], "o", mfc="white", mec=color(engine), ms=6,
            zorder=6)
    return bool((~valid).any())


def hollow_legend(ax) -> None:
    ax.plot([], [], "o", mfc="white", mec="gray", label="invalid or throttled point")


def slo_line(ax, value: float, orient: str = "h", label: str = "SLO") -> None:
    (ax.axhline if orient == "h" else ax.axvline)(value, ls="--", color="gray", lw=1, label=label)


def finish(fig, ctx: PlotContext, name: str, title: str, caption: str) -> Path:
    """Title, footer, watermarks and caption, then PNG and SVG into the run's plot folder."""
    fig.suptitle(f"{title}\n{ctx.title_suffix}", fontsize=10)
    versions = ", ".join(f"{k} {v}" for k, v in ctx.baseline_versions.items()) or "no baselines"
    locked = "clocks locked" if ctx.env.get("clocks_locked") else "clocks NOT locked"
    footer = (f"{ctx.run_dir.name} | engine {ctx.env.get('git_commit', '')[:7]} | {versions} | "
              f"{locked}")
    fig.text(0.01, 0.005, footer, fontsize=6, color="gray")
    fig.text(0.5, 0.045, caption, ha="center", fontsize=7, style="italic")
    if ctx.correctness_failed:
        fig.text(0.5, 0.5, "ENGINE FAILED CORRECTNESS", fontsize=28, color="red", alpha=0.25,
                 ha="center", va="center", rotation=25)
    if ctx.quick:
        fig.text(0.5, 0.35, "QUICK RUN", fontsize=28, color="gray", alpha=0.25, ha="center",
                 va="center", rotation=25)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    ctx.out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(ctx.out_dir / f"{name}.{suffix}", dpi=DPI)
    plt.close(fig)
    return ctx.out_dir / f"{name}.png"
