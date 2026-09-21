from __future__ import annotations

import numpy as np

from .style import color, figure, finish, ordered


def draw(ctx):
    data = ctx.table("coldstart")
    if data is None:
        return None
    fig, axes = figure(1, 2)
    engines = ordered(data.engine)
    for ax, column, label in ((axes[0][0], "wait_ready_s", "time to first served request (s)"),
                              (axes[0][1], "first_request_penalty_s", "first-request penalty (s)")):
        for k, cache in enumerate(("cold_cache", "warm_cache")):
            values = [data[(data.engine == e) & (data.cache == cache)][column].median()
                      for e in engines]
            ax.bar(np.arange(len(engines)) + k * 0.4, values, 0.4,
                   color=[color(e) for e in engines], alpha=1.0 if k == 0 else 0.5,
                   label=cache)
        ax.set_xticks(np.arange(len(engines)) + 0.2, engines)
        ax.set_ylabel(label)
    axes[0][0].legend(fontsize=8)
    return finish(fig, ctx, "p25_coldstart", "Cold start and first request",
                  "median over launches; the first launch follows a page-cache drop when permitted")
