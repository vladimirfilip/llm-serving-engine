from __future__ import annotations

import numpy as np

from .style import figure, finish

RANGES = ["schedule", "prepare_inputs", "forward", "sample", "postprocess"]


def draw(ctx):
    steps = ctx.table("nsys_steps")
    if steps is None:
        return None
    means = steps.groupby("batch").mean(numeric_only=True)
    fig, axes = figure(1, 2)
    x = np.arange(len(means))
    bottom = np.zeros(len(means))
    for name in RANGES:
        axes[0][0].bar(x, means[f"{name}_s"] * 1000, bottom=bottom, label=name)
        bottom += means[f"{name}_s"].to_numpy() * 1000
    axes[0][0].set_xticks(x, means.index)
    axes[0][0].set(xlabel="batch size", ylabel="mean time per step (ms)")
    axes[0][0].legend(fontsize=7)
    axes[0][1].bar(x, means.gpu_busy_s * 1000, label="GPU busy", color="#2ca02c")
    axes[0][1].bar(x, means.gpu_idle_s * 1000, bottom=means.gpu_busy_s * 1000, label="GPU idle",
                   color="#d62728")
    axes[0][1].set_xticks(x, means.index)
    axes[0][1].set(xlabel="batch size", ylabel="step period (ms)")
    axes[0][1].legend(fontsize=8)
    return finish(fig, ctx, "p18_ours_step_breakdown", "Ours: where an engine step goes",
                  "NVTX ranges on different threads overlap, so they need not sum to the period")
