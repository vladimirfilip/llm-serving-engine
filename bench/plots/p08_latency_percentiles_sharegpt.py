from __future__ import annotations

from .style import figure, finish, hollow_legend, ordered, points, to_ms

PERCENTILES = ("p50", "p95", "p99")


def draw(ctx):
    itl_ok = all(r["itl_valid"] for r in (ctx.json("checks.json") or {}).values()) \
        if ctx.json("checks.json") else True
    second = "itl" if itl_ok else "tpot"
    metrics = [f"{m}_{p}" for m in ("e2e", second) for p in PERCENTILES]
    data = ctx.sweep("sharegpt", metrics)
    if data is None:
        return None
    ms = to_ms(data, "e2e", second)
    fig, axes = figure(2, 3)
    hollow = False
    for row, family in enumerate(("e2e", second)):
        for col, p in enumerate(PERCENTILES):
            ax, metric = axes[row][col], f"{family}_{p}"
            for engine in ordered(ms.engine):
                hollow |= points(ax, engine, ms[ms.engine == engine], "offered_rps", metric)
            ax.set_yscale("log")
            ax.set_xlabel("offered load (requests/s)")
            ax.set_ylabel(f"{family} {p} (ms, log)")
    if hollow:
        hollow_legend(axes[0][0])
    axes[0][0].legend(fontsize=7)
    note = "" if itl_ok else "; ITL unavailable for an engine, TPOT shown"
    return finish(fig, ctx, "p08_latency_percentiles_sharegpt",
                  "ShareGPT: latency percentiles against offered load",
                  "end-to-end latency and inter-token latency, client-side" + note)
