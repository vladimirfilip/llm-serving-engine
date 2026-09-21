from __future__ import annotations

import numpy as np
import pandas as pd

from .style import color, figure, finish, ordered


def draw(ctx):
    data = ctx.table("nsys")
    if data is None:
        return None
    fig, axes = figure(1, 2)
    engines, batches = ordered(data.engine), sorted(data.batch.unique())
    width = 0.8 / len(engines)
    for i, engine in enumerate(engines):
        frame = data[data.engine == engine].set_index("batch").reindex(batches)
        axes[0][0].bar(np.arange(len(batches)) + i * width, frame.gpu_busy_fraction * 100, width,
                       color=color(engine), label=engine)
    axes[0][0].set_xticks(np.arange(len(batches)) + width * (len(engines) - 1) / 2, batches)
    axes[0][0].set(xlabel="batch size", ylabel="GPU busy (% of the capture window)")
    axes[0][0].legend(fontsize=8)
    gaps_path = ctx.run_dir / "nsys" / "gaps.parquet"
    if gaps_path.exists():
        gaps = pd.read_parquet(gaps_path)
        for engine in engines:
            g = gaps[(gaps.engine == engine) & (gaps.batch == 16)].gap_s * 1e6
            if len(g):
                axes[0][1].hist(g.clip(lower=0.1), bins=np.logspace(-1, 5, 60), histtype="step",
                                color=color(engine), lw=2, label=engine)
        axes[0][1].set(xscale="log", yscale="log", xlabel="gap between GPU work (us, log)",
                       ylabel="count (log)")
    return finish(fig, ctx, "p17_gpu_busy", "GPU busy fraction and idle gaps",
                  "Nsight Systems, union of kernel, copy and memset intervals; right: batch 16")
