from __future__ import annotations

from .style import figure, finish, hollow_legend, ordered, points, slo_line, to_ms

METRICS = ["out_tok_s", "tpot_p50", "tpot_p99"]


def draw(ctx):
    data = ctx.sweep("sharegpt", METRICS)
    if data is None:
        return None
    data = to_ms(data, "tpot")
    fig, axes = figure(1, 2)
    hollow = False
    for ax, metric in zip(axes[0], ("tpot_p50", "tpot_p99"), strict=True):
        for engine in ordered(data.engine):
            hollow |= points(ax, engine, data[data.engine == engine], "out_tok_s", metric)
        slo_line(ax, ctx.suite["slo"]["sharegpt"]["tpot_ms"], label="TPOT SLO")
        ax.set_yscale("log")
        ax.set_xlabel("output throughput (tokens/s)")
        ax.set_ylabel(f"{metric.replace('_', ' ')} (ms, log)")
    if hollow:
        hollow_legend(axes[0][0])
    axes[0][0].legend(fontsize=8)
    return finish(fig, ctx, "p01_throughput_latency_sharegpt",
                  "ShareGPT: throughput against TPOT",
                  "client-side timing, includes HTTP and detokenization; "
                  "one marker per offered rate")
