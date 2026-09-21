from __future__ import annotations

from .style import figure, finish, line, ordered


def draw(ctx):
    data = ctx.table("kernels_decode")
    if data is None:
        return None
    data = data[~data.dropped.astype(bool)]
    fig, axes = figure(1, 3)
    bw = ctx.env.get("bw_read_gbs")
    for ax, batch in zip(axes[0], (1, 16, 128), strict=True):
        ax.set_title(f"batch {batch}", fontsize=9)
        for contender in ordered(data.contender):
            frame = data[(data.contender == contender) & (data.batch == batch)].sort_values("ctx")
            if len(frame):
                line(ax, contender, frame.ctx, frame.gb_s)
        ax.set(xscale="log", xlabel="context (tokens, log)", ylabel="KV read bandwidth (GB/s)")
        if bw:
            twin = ax.twinx()
            twin.set_ylim(*(v / bw * 100 for v in ax.get_ylim()))
            twin.set_ylabel("% of measured read bandwidth")
    axes[0][0].legend(fontsize=8)
    return finish(fig, ctx, "p14_attn_decode", "Decode attention bandwidth",
                  "kernel only, L2 flushed, CUDA events; a batch that does not fit is absent")
