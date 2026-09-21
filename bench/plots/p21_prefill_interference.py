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
    fig, axes = figure(2, max(1, len(combos)))
    for col, (engine, variant) in enumerate(combos):
        frame = data[(data.engine == engine) & (data.variant == variant)]
        ax = axes[0][col]
        ax.plot(frame.t, frame.itl * 1000, ".", ms=1.5, color=color(engine), alpha=0.5)
        ax.set(yscale="log", title=f"{engine} / {variant}", xlabel="time (s)",
               ylabel="background ITL (ms, log)")
        low = axes[1][col]
        for label, style in (("baseline", "-"), ("during_prefill", "--")):
            x = np.sort(frame[frame.label == label].itl.to_numpy() * 1000)
            if len(x):
                low.plot(x, np.arange(1, len(x) + 1) / len(x), style, color=color(engine),
                         label=label)
        low.set(xscale="log", xlabel="ITL (ms, log)", ylabel="cumulative share")
        low.legend(fontsize=7)
    return finish(fig, ctx, "p21_prefill_interference", "Decode stalls during a long prefill",
                  "16 background streams; one 8192-token prompt injected every few seconds")
