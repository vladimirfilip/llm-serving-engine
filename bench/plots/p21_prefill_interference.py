from __future__ import annotations

import numpy as np
import pandas as pd

from .style import color, figure, finish


def load(ctx) -> pd.DataFrame | None:
    frames = []
    for path in sorted((ctx.run_dir / "scheduler").glob("*/interference_*.parquet")):
        frame = pd.read_parquet(path)
        frame["engine"] = path.parent.name
        frame["variant"] = path.stem.removeprefix("interference_")
        frames.append(frame)
    return pd.concat(frames) if frames else None


def draw(ctx):
    data = load(ctx)
    if data is None:
        return None
    combos = list(data[["engine", "variant"]].drop_duplicates().itertuples(index=False))
    # top: one time-series row per engine and variant; bottom: one CDF panel per combination
    fig, axes = figure(len(combos) + 1, max(1, len(combos)))
    for ax in axes.flat:
        ax.set_visible(False)
    for row, (engine, variant) in enumerate(combos):
        frame = data[(data.engine == engine) & (data.variant == variant)]
        series = axes[row][0]
        series.set_visible(True)
        series.plot(frame.t, frame.itl * 1000, ".", ms=1.5, color=color(engine), alpha=0.5)
        series.set(yscale="log", title=f"{engine} / {variant}", xlabel="time (s)",
                   ylabel="background ITL (ms, log)")
        cdf = axes[-1][row]
        cdf.set_visible(True)
        for label, style in (("baseline", "-"), ("during_prefill", "--")):
            x = np.sort(frame[frame.label == label].itl.to_numpy() * 1000)
            if len(x):
                cdf.plot(x, np.arange(1, len(x) + 1) / len(x), style, color=color(engine),
                         label=label)
        if not frame.label.notna().any():
            cdf.text(0.5, 0.5, "n/a (unlabelled)", transform=cdf.transAxes, ha="center")
        cdf.set(xscale="log", title=f"{engine} / {variant}", xlabel="ITL (ms, log)",
                ylabel="cumulative share")
        cdf.legend(fontsize=7)
    return finish(fig, ctx, "p21_prefill_interference", "Decode stalls during a long prefill",
                  "16 background streams; one long prompt injected every few seconds")
