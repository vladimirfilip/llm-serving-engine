from __future__ import annotations

import pandas as pd

from .style import figure, finish, line


def draw(ctx):
    folders = sorted((ctx.run_dir / "soak").glob("*/verdict.json"))
    if not folders:
        return None
    fig, axes = figure(2, 2)
    for verdict in folders:
        engine = verdict.parent.name
        memory = pd.read_csv(verdict.parent / "memory.csv")
        throughput = pd.read_csv(verdict.parent / "throughput.csv")
        latency = pd.read_csv(verdict.parent / "latency.csv")
        line(axes[0][0], engine, memory.t_s / 3600, memory.mem_used_bytes / 2**30, marker=None)
        line(axes[0][1], engine, memory.t_s / 3600, memory.server_rss_bytes / 2**30, marker=None)
        line(axes[1][0], engine, throughput.t_s / 3600, throughput.out_tok_s, marker=None)
        line(axes[1][1], engine, latency.t_s / 3600, latency.tpot_p99 * 1000, marker=None)
    for ax, label in zip(axes.flat, ("device memory (GiB)", "server RSS (GiB)",
                                     "output tokens/s per minute", "TPOT p99 per 5 min (ms)"),
                         strict=True):
        ax.set(xlabel="hours", ylabel=label)
    axes[0][0].legend(fontsize=8)
    return finish(fig, ctx, "p24_soak", "Long-run stability",
                  "open-loop ShareGPT at a fixed share of capacity")
