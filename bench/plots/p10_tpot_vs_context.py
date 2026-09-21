from __future__ import annotations

from .style import figure, finish, line, ordered


def draw(ctx):
    data = ctx.table("single_decode")
    if data is None:
        return None
    fig, axes = figure()
    ax = axes[0][0]
    for engine in ordered(data.engine):
        frame = data[data.engine == engine].sort_values("context")
        line(ax, engine, frame.context, frame.tpot_s * 1000)
    bound = data.dropna(subset=["bound_tok_s"]).drop_duplicates("context").sort_values("context")
    if len(bound):
        ax.plot(bound.context, 1000 / bound.bound_tok_s, "--", color="gray",
                label="bandwidth bound")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("context length (tokens, log2)")
    ax.set_ylabel("TPOT (ms)")
    ax.legend(fontsize=8)
    return finish(fig, ctx, "p10_tpot_vs_context", "Decode latency against context length",
                  "through the server at concurrency 1; dashed: TPOT the measured bandwidth allows")
