from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from .style import figure, finish, ordered

CODES = {"ok": 0, "preempted": 1, "failed": 2, "not run": 3}
PALETTE = ["#2ca02c", "#ffbf00", "#d62728", "#cccccc"]


def draw(ctx):
    data = ctx.table("memory_grid")
    if data is None:
        return None
    engines = ordered(data.engine)
    fig, axes = figure(1, len(engines))
    for ax, engine in zip(axes[0], engines, strict=True):
        frame = data[data.engine == engine]
        lengths, batches = sorted(frame.length.unique()), sorted(frame.batch.unique())
        grid = np.full((len(batches), len(lengths)), 3)
        for _, r in frame.iterrows():
            i, j = batches.index(r.batch), lengths.index(r.length)
            grid[i, j] = CODES[r.status]
            ax.text(j, i, "n/a" if r.peak_running != r.peak_running else f"{int(r.peak_running)}",
                    ha="center", va="center", fontsize=7)
        ax.imshow(grid, cmap=ListedColormap(PALETTE), vmin=0, vmax=3, aspect="auto",
                  origin="lower")
        ax.set_xticks(range(len(lengths)), lengths, fontsize=7)
        ax.set_yticks(range(len(batches)), batches, fontsize=7)
        ax.set(title=engine, xlabel="length (tokens)", ylabel="batch size")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in PALETTE]
    axes[0][0].legend(handles, list(CODES), fontsize=7, loc="upper left")
    return finish(fig, ctx, "p19_capacity_heatmap", "Batch and length the engine holds",
                  "cell text: most sequences the engine ran together; n/a where it has no metric")
