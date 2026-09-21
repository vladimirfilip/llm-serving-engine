from __future__ import annotations

from .style import figure, finish, hollow_legend, ordered, points, slo_line, to_ms

WORKLOADS = ["short_short", "long_short", "short_long", "ctx32k"]


def draw(ctx):
    panels = {w: ctx.sweep(w, ["out_tok_s", "tpot_p99"]) for w in WORKLOADS}
    if all(d is None for d in panels.values()):
        return None
    fig, axes = figure(2, 2)
    hollow = False
    for ax, workload in zip(axes.flat, WORKLOADS, strict=True):
        data = panels[workload]
        ax.set_title(workload, fontsize=9)
        if data is None:
            ax.text(0.5, 0.5, "not run", ha="center", va="center", transform=ax.transAxes)
            continue
        ms = to_ms(data, "tpot")
        for engine in ordered(ms.engine):
            hollow |= points(ax, engine, ms[ms.engine == engine], "out_tok_s", "tpot_p99")
        slo_line(ax, ctx.suite["slo"][workload]["tpot_ms"], label="TPOT SLO")
        ax.set_yscale("log")
        ax.set_xlabel("output throughput (tokens/s)")
        ax.set_ylabel("TPOT p99 (ms, log)")
    if hollow:
        hollow_legend(axes[0][0])
    axes[0][0].legend(fontsize=7)
    return finish(fig, ctx, "p09_fixed_shapes",
                  "Fixed-shape workloads: throughput against p99 TPOT",
                  "client-side timing; a workload the model length cannot run is marked not run")
