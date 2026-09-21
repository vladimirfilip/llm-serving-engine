from __future__ import annotations

from .style import figure, finish, hollow_legend, ordered, points


def draw(ctx):
    data = ctx.sweep("sharegpt", ["energy_j_per_out_token"])
    if data is None or data.energy_j_per_out_token.isna().all():
        return None
    data = data.dropna(subset=["energy_j_per_out_token"]).copy()
    for suffix in ("", "_min", "_max"):
        data[f"tokens_per_joule{suffix}"] = 1 / data[f"energy_j_per_out_token{suffix}"]
    data = data.rename(columns={"tokens_per_joule_min": "tokens_per_joule_max_",
                                "tokens_per_joule_max": "tokens_per_joule_min"}).rename(
        columns={"tokens_per_joule_max_": "tokens_per_joule_max"})
    fig, axes = figure()
    hollow = False
    for engine in ordered(data.engine):
        hollow |= points(axes[0][0], engine, data[data.engine == engine], "offered_rps",
                         "tokens_per_joule")
    if hollow:
        hollow_legend(axes[0][0])
    axes[0][0].set(xlabel="offered load (requests/s)", ylabel="output tokens per joule")
    axes[0][0].legend(fontsize=8)
    return finish(fig, ctx, "p23_energy", "ShareGPT: output tokens per joule",
                  "NVML total energy counter over the measurement window, whole board")
