from __future__ import annotations

import numpy as np
import pandas as pd

from .style import figure, finish


def draw(ctx):
    paths = sorted((ctx.run_dir / "scheduler").glob("*/overload.parquet"))
    if not paths:
        return None
    metrics = ctx.table("scheduler_overload")
    fig, axes = figure(1, len(paths))
    for ax, path in zip(axes[0], paths, strict=True):
        engine = path.parent.name
        frame = pd.read_parquet(path)
        frame = frame[frame.status == "ok"]
        terciles = pd.qcut(frame.prompt_len, 3, labels=["short", "medium", "long"],
                           duplicates="drop")
        for label, colour in zip(terciles.cat.categories, ("#1f77b4", "#ff7f0e", "#d62728"),
                                 strict=False):
            chosen = terciles == label
            ttft = np.maximum(frame.t_first - frame.t_send, 1e-3)
            ax.scatter(frame.t_sched[chosen], ttft[chosen], s=6, color=colour,
                       label=f"{label} prompts")
        ax.set(yscale="log", title=engine, xlabel="scheduled send time (s)",
               ylabel="TTFT (s, log)")
        if metrics is not None:
            rho = metrics[metrics.engine == engine].spearman_prompt_len_ttft
            if len(rho):
                ax.text(0.02, 0.95, f"Spearman(prompt len, TTFT) = {rho.iloc[0]:.2f}",
                        transform=ax.transAxes, fontsize=8, va="top")
        ax.legend(fontsize=7, loc="lower right")
    return finish(fig, ctx, "p22_overload_ttft", "Who waits under overload",
                  "open-loop ShareGPT above capacity; colour: prompt-length tercile")
