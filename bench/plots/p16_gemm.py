from __future__ import annotations

import numpy as np
from matplotlib.colors import TwoSlopeNorm

from .style import figure, finish, line


def draw(ctx):
    data = ctx.table("kernels_gemm")
    if data is None:
        return None
    ours, torch_ = (data[data.contender == c].set_index(["shape", "m"]) for c in ("ours", "torch"))
    ratio = (ours.ms_median / torch_.ms_median).unstack("m")
    fig, axes = figure(1, 2)
    ax = axes[0][0]
    log_ratio = np.log2(ratio.to_numpy())
    span = max(abs(log_ratio).max(), 0.01)
    image = ax.imshow(log_ratio, cmap="RdBu_r", aspect="auto",
                      norm=TwoSlopeNorm(0.0, vmin=-span, vmax=span))
    ax.set_xticks(range(ratio.shape[1]), [str(m) for m in ratio.columns], fontsize=7)
    ax.set_yticks(range(ratio.shape[0]), ratio.index, fontsize=8)
    ax.set_xlabel("M (rows)")
    for i, j in np.ndindex(ratio.shape):
        ax.text(j, i, f"{ratio.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(image, ax=ax, label="log2(t_ours / t_torch); 0 is parity")
    small = data[data.m <= 64]
    for shape in ("qkv_proj", "down_proj"):
        for contender in ("ours", "torch"):
            chosen = (small.contender == contender) & (small["shape"] == shape)
            frame = small[chosen].sort_values("m")
            line(axes[0][1], contender, frame.m, frame.gb_s,
                 ls="-" if shape == "qkv_proj" else "--", label=f"{contender} {shape}")
    if ctx.env.get("bw_read_gbs"):
        axes[0][1].axhline(ctx.env["bw_read_gbs"], color="gray", ls=":", label="measured read bw")
    axes[0][1].set(xscale="log", xlabel="M (log)", ylabel="GB/s")
    axes[0][1].legend(fontsize=6)
    return finish(fig, ctx, "p16_gemm", "Projection GEMMs, ours against torch",
                  "ours runs separate q, k, v and gate, up linears; torch one fused matmul")
