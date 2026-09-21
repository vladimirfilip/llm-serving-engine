from __future__ import annotations

import numpy as np

from .style import color, figure, finish, ordered


def draw(ctx):
    data = ctx.table("single_decode")
    if data is None or data.bound_fraction.isna().all():
        return None
    fig, axes = figure()
    ax = axes[0][0]
    contexts = sorted(data.context.unique())
    engines = ordered(data.engine)
    width = 0.8 / len(engines)
    for i, engine in enumerate(engines):
        frame = data[data.engine == engine].set_index("context").reindex(contexts)
        x = np.arange(len(contexts)) + i * width
        bars = ax.bar(x, frame.bound_fraction * 100, width, color=color(engine), label=engine)
        for bar, tpot in zip(bars, frame.tpot_s, strict=True):
            if tpot == tpot:
                ax.annotate(f"{tpot * 1000:.1f}", (bar.get_x() + bar.get_width() / 2,
                                                   bar.get_height()), ha="center", fontsize=6,
                            xytext=(0, 2), textcoords="offset points")
    ax.axhline(100, ls="--", color="gray", lw=1, label="bandwidth bound")
    ax.set_xticks(np.arange(len(contexts)) + width * (len(engines) - 1) / 2, contexts)
    ax.set_xlabel("context length (tokens)")
    ax.set_ylabel("share of the bandwidth-bound decode rate (%)")
    ax.legend(fontsize=8)
    return finish(fig, ctx, "p02_batch1_bound_fraction",
                  "Batch-1 decode against the bandwidth bound",
                  "through the server, includes HTTP and detokenization; bar labels: TPOT in ms")
