from __future__ import annotations

from .style import color, figure, finish, line, ordered


def draw(ctx):
    data = ctx.table("single_batch")
    if data is None:
        return None
    fig, axes = figure(1, 2)
    for engine in ordered(data.engine):
        frame = data[data.engine == engine].sort_values("batch")
        line(axes[0][0], engine, frame.batch, frame.out_tok_s)
        line(axes[0][1], engine, frame.batch, frame.per_stream_tok_s)
        saturation = frame.saturation_batch.dropna()
        if len(saturation):
            at = frame[frame.batch == saturation.iloc[0]]
            axes[0][0].plot(at.batch, at.out_tok_s, "*", ms=14, color=color(engine), mec="white")
    for ax, label in zip(axes[0], ("output throughput (tokens/s)", "per-stream tokens/s"),
                         strict=True):
        ax.set(xscale="log", xlabel="batch size (closed-loop streams, log2)", ylabel=label)
        ax.set_xscale("log", base=2)
    axes[0][0].legend(fontsize=8)
    return finish(fig, ctx, "p12_throughput_vs_batch", "Closed-loop throughput against batch size",
                  "prompt 128, output 512; star: first batch size whose doubling gains under 5%")
