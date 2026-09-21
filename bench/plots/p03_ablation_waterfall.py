from __future__ import annotations

import numpy as np

from .style import figure, finish


def draw(ctx):
    data = ctx.table("ablation")
    if data is None:
        return None
    data = data[data.skipped.isna()] if "skipped" in data else data
    fig, axes = figure(1, 2)
    for ax, metric, label in ((axes[0][0], "out_tok_s", "ShareGPT capacity (tokens/s)"),
                              (axes[0][1], "tpot_s", "batch-1 TPOT (ms)")):
        scale = 1000 if metric == "tpot_s" else 1
        values = data[metric].to_numpy() * scale
        lo = values - data[f"{metric}_min"].to_numpy() * scale
        hi = data[f"{metric}_max"].to_numpy() * scale - values
        x = np.arange(len(data))
        ax.bar(x, values, yerr=[lo.clip(0), hi.clip(0)], color="#111111", capsize=3)
        for i in range(len(values)):
            if i:
                ax.annotate(f"{100 * (values[i] / values[i - 1] - 1):+.0f}%", (x[i], values[i]),
                            ha="center", fontsize=8, xytext=(0, 4), textcoords="offset points")
        ax.set_xticks(x, data.step, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel(label)
    return finish(fig, ctx, "p03_ablation_waterfall",
                  "Ablation: ours, optimization by optimization",
                  "cumulative steps; error bars: min to max across repeats; label: change from the "
                  "previous step")
