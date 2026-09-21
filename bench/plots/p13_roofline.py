from __future__ import annotations

import numpy as np

from ..modelspec import ModelSpec
from .style import figure, finish

POINTS = {"decode": [(1, 8192), (16, 8192), (128, 8192)], "prefill": [2048, 8192],
          "gemm_m": [1, 16, 256, 2048]}


def draw(ctx):
    decode, prefill, gemm = (ctx.table(n) for n in ("kernels_decode", "kernels_prefill",
                                                    "kernels_gemm"))
    env = ctx.env
    if decode is None and prefill is None and gemm is None:
        return None
    spec = ModelSpec.from_config(
        {"hidden_size": env["model"]["spec"]["d"], "intermediate_size": env["model"]["spec"]["f"],
         "num_hidden_layers": env["model"]["spec"]["n_layers"],
         "num_attention_heads": env["model"]["spec"]["n_heads"],
         "num_key_value_heads": env["model"]["spec"]["n_kv_heads"],
         "head_dim": env["model"]["spec"]["head_dim"], "vocab_size": env["model"]["spec"]["vocab"]},
        env["model"]["dtype"])
    bw, peak = env.get("bw_read_gbs"), env["hardware"]["peak_tflops_bf16_dense"]
    fig, axes = figure()
    ax = axes[0][0]
    xs = np.logspace(-1, 4, 50)
    if bw:
        ax.plot(xs, bw * xs / 1000, "--", color="gray", label=f"{bw:.0f} GB/s (measured read)")
    ax.axhline(peak, ls=":", color="gray", label=f"{peak} TFLOP/s (datasheet peak)")
    ours = lambda d: d[d.contender == "ours"] if d is not None else None  # noqa: E731
    if ours(decode) is not None:
        for b, c in POINTS["decode"]:
            r = ours(decode)[(ours(decode).batch == b) & (ours(decode).ctx == c)]
            if len(r):
                flops = 4 * b * c * spec.n_heads * spec.head_dim
                moved = b * c * spec.kv_bytes_per_tok_layer
                ax.plot(flops / moved, flops / (r.ms_median.iloc[0] / 1000) / 1e12, "o",
                        label=f"decode attn B={b}")
    if ours(prefill) is not None:
        for length in POINTS["prefill"]:
            r = ours(prefill)[ours(prefill).seq_len == length]
            if len(r):
                flops = 2 * length**2 * spec.n_heads * spec.head_dim
                moved = 2 * length * (spec.n_heads + 2 * spec.n_kv_heads) * spec.head_dim \
                    * spec.dtype_bytes
                ax.plot(flops / moved, r.tflop_s.iloc[0], "s", label=f"prefill attn L={length}")
    if ours(gemm) is not None:
        for m in POINTS["gemm_m"]:
            for _, r in ours(gemm)[ours(gemm).m == m].iterrows():
                intensity = 2 * m * r.n * r.k / ((r.n * r.k + m * (r.n + r.k)) * spec.dtype_bytes)
                ax.plot(intensity, r.tflop_s, "^", color="#111111", alpha=0.5)
    ax.set(xscale="log", yscale="log", xlabel="arithmetic intensity (FLOP/byte, log)",
           ylabel="TFLOP/s (log)")
    ax.legend(fontsize=6)
    return finish(fig, ctx, "p13_roofline", "Roofline of ours' kernels",
                  "kernel only, L2 flushed, CUDA events; triangles: projection GEMMs")
