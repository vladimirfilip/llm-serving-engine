from __future__ import annotations

import numpy as np

from .style import color, figure, finish, ordered

GIB = 2**30


def draw(ctx):
    breakdown, device = ctx.table("memory_breakdown"), ctx.table("memory_device")
    if breakdown is None and device is None:
        return None
    fig, axes = figure(1, 2)
    if breakdown is not None:
        ours = breakdown[breakdown.engine == "ours"]
        phases = [p for p in ("idle", "loaded") if p in set(ours.phase)]
        bottom = np.zeros(len(phases))
        for component in ours.component.unique():
            heights = np.array([ours[(ours.phase == p) & (ours.component == component)].bytes.sum()
                                for p in phases]) / GIB
            axes[0][0].bar(phases, heights, bottom=bottom, label=component)
            bottom += heights
        axes[0][0].set_ylabel("GiB")
        axes[0][0].legend(fontsize=7)
        axes[0][0].set_title("ours, from the engine's allocator accounting", fontsize=9)
    if device is not None:
        engines = ordered(device.engine)
        x = np.arange(len(engines))
        frame = device.set_index("engine").reindex(engines)
        axes[0][1].bar(x - 0.2, frame.idle_bytes / GIB, 0.4, label="idle", color="#999999")
        axes[0][1].bar(x + 0.2, frame.peak_bytes / GIB, 0.4, label="peak under load",
                       color=[color(e) for e in engines])
        axes[0][1].set_xticks(x, engines)
        axes[0][1].set_ylabel("device memory used (GiB, NVML)")
        axes[0][1].legend(fontsize=8)
    return finish(fig, ctx, "p04_memory_breakdown", "Device memory: breakdown and peaks",
                  "left: ours only; right: NVML device memory for every engine")
