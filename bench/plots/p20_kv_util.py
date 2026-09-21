from __future__ import annotations

from .style import figure, finish, line, ordered


def draw(ctx):
    data = ctx.table("kv_util")
    if data is None:
        return None
    checks = ctx.json("checks.json") or {}
    fig, axes = figure(1, 2)
    for engine, c in checks.items():
        if not c.get("stats_available", True):
            for ax in axes[0]:
                ax.text(0.5, 0.9 - 0.06 * list(checks).index(engine), f"{engine}: n/a (no stats)",
                        transform=ax.transAxes, ha="center", fontsize=8)
    for engine in ordered(data.engine):
        frame = data[data.engine == engine].sort_values("t_s")
        line(axes[0][0], engine, frame.t_s, frame.kv_util * 100, marker=None)
        if frame.internal_frag.notna().any():
            line(axes[0][1], engine, frame.t_s, frame.internal_frag * 100, marker=None)
        else:
            axes[0][1].text(0.5, 0.5, f"{engine}: n/a", transform=axes[0][1].transAxes,
                            ha="center", fontsize=8)
    axes[0][0].set(xlabel="time (s)", ylabel="KV pool in use (%)")
    axes[0][1].set(xlabel="time (s)", ylabel="internal fragmentation (%)")
    axes[0][0].legend(fontsize=8)
    return finish(fig, ctx, "p20_kv_util", "KV utilisation and internal fragmentation",
                  "open-loop ShareGPT at a fixed share of the reference capacity")
