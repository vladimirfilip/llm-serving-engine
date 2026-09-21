from __future__ import annotations

from .style import figure, finish, hollow_legend, ordered, points, slo_line, to_ms

METRICS = ["ttft_p50", "ttft_p99"]


def draw(ctx):
    data = ctx.sweep("sharegpt", METRICS)
    if data is None:
        return None
    ms = to_ms(data, "ttft")
    fig, axes = figure(1, 2)
    hollow = False
    for ax, metric in zip(axes[0], METRICS, strict=True):
        for engine in ordered(ms.engine):
            hollow |= points(ax, engine, ms[ms.engine == engine], "offered_rps", metric)
        slo_line(ax, ctx.suite["slo"]["sharegpt"]["ttft_ms"], label="TTFT SLO")
        ax.set_yscale("log")
        ax.set_xlabel("offered load (requests/s)")
        ax.set_ylabel(f"{metric.replace('_', ' ')} (ms, log)")
    if hollow:
        hollow_legend(axes[0][0])
    axes[0][0].legend(fontsize=8)
    return finish(fig, ctx, "p07_ttft_vs_rate_sharegpt", "ShareGPT: TTFT against offered load",
                  "client-side timing, includes HTTP and tokenization")
