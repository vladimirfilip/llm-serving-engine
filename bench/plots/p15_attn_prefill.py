from __future__ import annotations

from .style import figure, finish, line, ordered


def draw(ctx):
    data = ctx.table("kernels_prefill")
    if data is None:
        return None
    data = data[~data.dropped.astype(bool)]
    fig, axes = figure()
    ax = axes[0][0]
    for contender in ordered(data.contender):
        frame = data[data.contender == contender].sort_values("seq_len")
        line(ax, contender, frame.seq_len, frame.tflop_s)
    ax.axhline(ctx.env["hardware"]["peak_tflops_bf16_dense"], ls="--", color="gray",
               label="datasheet peak")
    ax.set(xscale="log", xlabel="sequence length (tokens, log)", ylabel="TFLOP/s")
    ax.legend(fontsize=8)
    return finish(fig, ctx, "p15_attn_prefill", "Prefill attention throughput",
                  "causal, batch 1, kernel only, L2 flushed, CUDA events")
