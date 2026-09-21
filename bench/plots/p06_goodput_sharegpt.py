from __future__ import annotations

from .style import color, figure, finish, hollow_legend, ordered, points

METRICS = ["goodput_rps"]


def draw(ctx):
    data = ctx.sweep("sharegpt", METRICS)
    if data is None:
        return None
    fig, axes = figure()
    ax = axes[0][0]
    hollow = False
    top = data.offered_rps.max()
    ax.plot([0, top], [0, top], "--", color="gray", lw=1, label="goodput = offered")
    summary = ctx.table("sweep_summary")
    for engine in ordered(data.engine):
        hollow |= points(ax, engine, data[data.engine == engine], "offered_rps", "goodput_rps")
        if summary is not None:
            row = summary[(summary.engine == engine) & (summary.workload == "sharegpt")]
            if len(row) and row.max_sustainable_rps.notna().iloc[0]:
                rate = row.max_sustainable_rps.iloc[0]
                ax.axvline(rate, color=color(engine), ls=":", lw=1)
                at = data[(data.engine == engine) & (data.offered_rps == rate)]
                ax.plot(at.offered_rps, at.goodput_rps, "D", color=color(engine), ms=8,
                        mec="white", zorder=7)
    if hollow:
        hollow_legend(ax)
    ax.set_xlabel("offered load (requests/s)")
    ax.set_ylabel("goodput (requests/s meeting the SLO)")
    ax.legend(fontsize=8)
    return finish(fig, ctx, "p06_goodput_sharegpt", "ShareGPT: goodput against offered load",
                  "diamond and dotted line: largest rate with at least 90% SLO attainment")
