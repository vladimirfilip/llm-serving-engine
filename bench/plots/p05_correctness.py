from __future__ import annotations

import numpy as np

from ..reference import read_jsonl
from .style import color, figure, finish, ordered


def logprob_differences(run_dir, engine: str) -> np.ndarray | None:
    folder = run_dir / "correctness" / engine
    if not (folder / "gen.jsonl").exists() or not (folder / "scored.jsonl").exists():
        return None
    ref = {r["id"]: r["ref_logprob"] for r in read_jsonl(folder / "scored.jsonl")}
    diffs = [np.abs(np.asarray(g["logprobs"]) - np.asarray(ref[g["id"]]))
             for g in read_jsonl(folder / "gen.jsonl") if g["id"] in ref
             and len(g["logprobs"]) == len(ref[g["id"]])]
    return np.concatenate(diffs) if diffs else None


def draw(ctx):
    summary = ctx.json("correctness/summary.json")
    if summary is None:
        return None
    fig, axes = figure(1, 2)
    engines = ordered(e for e in summary if e != "gates")
    for engine in engines:
        diffs = logprob_differences(ctx.run_dir, engine)
        if diffs is not None and len(diffs):
            x = np.sort(np.maximum(diffs, 1e-6))
            axes[0][0].plot(x, np.arange(1, len(x) + 1) / len(x), color=color(engine),
                            lw=3 if engine == "ours" else 1.6, label=engine)
    axes[0][0].set(xscale="log", xlabel="|engine logprob - reference logprob| (nats, log)",
                   ylabel="cumulative share of positions")
    axes[0][0].legend(fontsize=8)
    with_ppl = [e for e in engines if summary[e].get("ppl") and summary[e].get("ref_ppl")]
    axes[0][1].bar(with_ppl, [summary[e]["ppl"] / summary[e]["ref_ppl"] for e in with_ppl],
                   color=[color(e) for e in with_ppl])
    axes[0][1].axhline(1.0, ls="--", color="gray", lw=1)
    axes[0][1].set_ylabel("perplexity / reference perplexity")
    for engine in engines:
        if engine not in with_ppl:
            axes[0][1].text(0.5, 0.5, f"{engine}: perplexity n/a", transform=axes[0][1].transAxes,
                            ha="center", fontsize=8)
    return finish(fig, ctx, "p05_correctness",
                  "Logprob agreement and perplexity against the reference",
                  "teacher-forced on each engine's own greedy tokens; WikiText-2 test windows")
