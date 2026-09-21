from __future__ import annotations

from .style import figure, finish, line, ordered


def draw(ctx):
    data = ctx.table("single_ttft")
    if data is None:
        return None
    fig, axes = figure(1, 2)
    for engine in ordered(data.engine):
        frame = data[data.engine == engine].sort_values("prompt")
        line(axes[0][0], engine, frame.prompt, frame.ttft_s * 1000)
        line(axes[0][1], engine, frame.prompt, frame.prefill_mfu * 100)
    axes[0][0].set(xscale="log", yscale="log", xlabel="prompt length (tokens, log)",
                   ylabel="TTFT (ms, log)")
    axes[0][1].set(xscale="log", xlabel="prompt length (tokens, log)",
                   ylabel="prefill MFU (% of datasheet peak)")
    axes[0][0].legend(fontsize=8)
    return finish(fig, ctx, "p11_ttft_vs_prompt", "TTFT and prefill MFU against prompt length",
                  "through the server; MFU is against the datasheet peak, not the clock-locked one")
