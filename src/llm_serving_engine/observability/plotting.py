"""Benchmark plots, drawn from the raw per-request results the load generator writes.
matplotlib is a dev dependency, so it is imported lazily.

Conventions shared by every load plot:
- A marker's fill is its point's keep-up verdict: filled kept up, hollow fell behind (its
  values depend on run length), x too few requests to tell.
- A rate measured over repeated runs is drawn at its median, with a bar from min to max.
- A latency percentile over too few samples to differ from the maximum is not drawn.
"""

from __future__ import annotations

from statistics import median

from ..loadgen.report import LatencyReport
from .metrics import LatencySummary, min_samples, percentile

KEEPS_UP_NOTE = "filled: kept up   hollow: fell behind   x: too few requests to tell"
_PERCENTILES = {"p50": 50, "p95": 95, "p99": 99}


def trusted_ms(summary: LatencySummary | None, name: str) -> float | None:
    """`summary`'s percentile `name` ("p50", "p95" or "p99") in milliseconds, or None if it
    rests on too few samples to be anything but the maximum."""
    if summary is None or summary.count < min_samples(_PERCENTILES[name]):
        return None
    return getattr(summary, name) * 1000


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _figure(xlabel: str, ylabel: str, title: str | None = None):
    fig, ax = _pyplot().subplots()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    return fig, ax


def _save(fig, out_path: str, keeps_up_note: bool = True) -> None:
    if keeps_up_note:
        fig.tight_layout(rect=(0, 0.04, 1, 1))  # the strip below the axes holds the note
        fig.text(0.01, 0.01, KEEPS_UP_NOTE, fontsize="x-small", color="gray")
    else:
        fig.tight_layout()
    fig.savefig(out_path)
    _pyplot().close(fig)


def _mark(ax, x: float, y: float, keeps_up: bool | None, color) -> None:
    if keeps_up is None:
        ax.plot(x, y, marker="x", color=color, linestyle="none")
    else:
        face = color if keeps_up else "none"
        ax.plot(x, y, marker="o", color=color, markerfacecolor=face, linestyle="none")


def _draw_series(
    ax, xs: list[float], ys: list[float | None], keeps_up: list[bool | None], **line_kwargs
):
    """A line through the points whose y is not None, marked by keep-up verdict. Returns the
    line's color, or None if every y was None."""
    points = [(x, y, k) for x, y, k in zip(xs, ys, keeps_up, strict=True) if y is not None]
    if not points:
        return None
    (line,) = ax.plot([p[0] for p in points], [p[1] for p in points], **line_kwargs)
    for x, y, k in points:
        _mark(ax, x, y, k, line.get_color())
    return line.get_color()


def _draw_spread(
    ax, xs: list[float], repeats: list[list[float | None]], keeps_up: list[bool | None], **kwargs
) -> None:
    """Median line over each point's repeats, with a min-to-max bar."""
    values = [[v for v in point if v is not None] for point in repeats]
    medians = [median(v) if v else None for v in values]
    color = _draw_series(ax, xs, medians, keeps_up, **kwargs)
    for x, v, m in zip(xs, values, medians, strict=True):
        if len(v) > 1:
            ax.errorbar(x, m, yerr=[[m - min(v)], [max(v) - m]], color=color, capsize=3)


def plot_latency_pareto(
    throughputs: list[list[float]],
    latencies: list[LatencyReport],
    keeps_up: list[bool | None],
    out_path: str,
) -> None:
    """End-to-end latency p50 (solid) and p99 (dashed) per request shape against median
    achieved throughput, one point per offered load. Shapes are never blended: a shape's
    output length sets most of its latency, so a blend would chart the workload mix."""
    fig, ax = _figure(
        "achieved throughput (req/s, median of repeats)", "end-to-end latency (ms)",
        "Latency by request shape vs throughput",
    )
    xs = [median(t) for t in throughputs]
    shapes = sorted({shape for latency in latencies for shape in latency.e2e_by_shape})
    for shape in shapes:
        summaries = [latency.e2e_by_shape.get(shape) for latency in latencies]
        color = _draw_series(
            ax, xs, [trusted_ms(s, "p50") for s in summaries], keeps_up, label=f"{shape} p50"
        )
        _draw_series(
            ax, xs, [trusted_ms(s, "p99") for s in summaries], keeps_up,
            label=f"{shape} p99", linestyle="--", color=color,
        )
    ax.set_yscale("log")
    ax.legend(fontsize="small")
    _save(fig, out_path)


def _plot_percentiles_by_load(
    load_values: list[float],
    summaries: list[LatencySummary | None],
    keeps_up: list[bool | None],
    out_path: str,
    ylabel: str,
    title: str,
) -> None:
    fig, ax = _figure("offered request rate (req/s)", ylabel, title)
    for name in _PERCENTILES:
        _draw_series(
            ax, load_values, [trusted_ms(s, name) for s in summaries], keeps_up, label=name
        )
    ax.legend()
    _save(fig, out_path)


def plot_ttft_by_qps(
    qps_values: list[float],
    ttft_by_shape: list[dict[str, LatencySummary]],
    keeps_up: list[bool | None],
    out_path: str,
) -> None:
    """Offered request rate vs TTFT p50 (solid) and p99 (dashed), one color per request
    shape, on a log axis: TTFT grows with prompt length, so shapes span orders of magnitude.
    Each legend entry carries the shape's fewest and most requests at any point."""
    fig, ax = _figure("offered request rate (req/s)", "TTFT (ms)", "TTFT by request shape")
    shapes = sorted({shape for point in ttft_by_shape for shape in point})
    for shape in shapes:
        summaries = [point.get(shape) for point in ttft_by_shape]
        counts = [s.count for s in summaries if s is not None]
        label = f"{shape} (n {min(counts)}-{max(counts)})"
        color = _draw_series(
            ax, qps_values, [trusted_ms(s, "p50") for s in summaries], keeps_up,
            label=f"{label} p50",
        )
        _draw_series(
            ax, qps_values, [trusted_ms(s, "p99") for s in summaries], keeps_up,
            label=f"{shape} p99", linestyle="--", color=color,
        )
    ax.set_yscale("log")
    ax.legend(fontsize="x-small")
    _save(fig, out_path)


def plot_tpot_by_qps(
    qps_values: list[float],
    tpot_summaries: list[LatencySummary | None],
    keeps_up: list[bool | None],
    out_path: str,
) -> None:
    """Offered request rate vs each request's mean time per output token. A mean over one
    request hides its stalls; plot_itl_by_qps shows those."""
    _plot_percentiles_by_load(
        qps_values, tpot_summaries, keeps_up, out_path, "TPOT (ms)", "Time per output token"
    )


def plot_itl_by_qps(
    qps_values: list[float],
    itl_summaries: list[LatencySummary | None],
    keeps_up: list[bool | None],
    out_path: str,
) -> None:
    """Offered request rate vs every gap between consecutive tokens, across all requests:
    the stall a client actually sees mid-stream."""
    _plot_percentiles_by_load(
        qps_values, itl_summaries, keeps_up, out_path, "inter-token latency (ms)",
        "Inter-token latency",
    )


def plot_e2e_latency_by_qps(
    qps_values: list[float],
    e2e_summaries: list[LatencySummary | None],
    keeps_up: list[bool | None],
    out_path: str,
) -> None:
    """Offered request rate vs end-to-end latency over every request shape."""
    _plot_percentiles_by_load(
        qps_values, e2e_summaries, keeps_up, out_path, "end-to-end latency (ms)",
        "End-to-end latency, all shapes",
    )


def plot_throughput_by_qps(
    qps_values: list[float],
    throughputs: list[list[float]],
    offered: list[list[float]],
    keeps_up: list[bool | None],
    out_path: str,
) -> None:
    """Target request rate vs achieved throughput, beside the arrival rate each run actually
    drew: a Poisson schedule over a short run lands off its target, and a server that keeps
    up tracks its arrivals, not the target."""
    fig, ax = _figure("target request rate (req/s)", "request rate (req/s)", "Throughput")
    _draw_spread(ax, qps_values, offered, keeps_up, label="arrivals", linestyle="--")
    _draw_spread(ax, qps_values, throughputs, keeps_up, label="achieved throughput")
    ax.legend()
    _save(fig, out_path)


def _plot_rate_by_load(
    load_values: list[float],
    repeats: list[list[float | None]],
    keeps_up: list[bool | None],
    out_path: str,
    ylabel: str,
    title: str,
) -> None:
    fig, ax = _figure("offered request rate (req/s)", ylabel, title)
    _draw_spread(ax, load_values, repeats, keeps_up)
    _save(fig, out_path)


def plot_output_throughput_by_qps(
    qps_values: list[float],
    output_tokens_s: list[list[float]],
    keeps_up: list[bool | None],
    out_path: str,
) -> None:
    _plot_rate_by_load(
        qps_values, output_tokens_s, keeps_up, out_path,
        "output-token throughput (tok/s)", "Output-token throughput",
    )


def plot_goodput_by_qps(
    qps_values: list[float],
    goodputs: list[list[float | None]],
    keeps_up: list[bool | None],
    out_path: str,
) -> None:
    """Requests/s meeting every configured SLO."""
    _plot_rate_by_load(qps_values, goodputs, keeps_up, out_path, "goodput (req/s)", "Goodput")


def plot_gpu_memory_by_load(
    load_values: list[float],
    peak_memory_mb: list[list[float | None]],
    keeps_up: list[bool | None],
    out_path: str,
) -> None:
    _plot_rate_by_load(
        load_values, peak_memory_mb, keeps_up, out_path, "peak GPU memory (MB)",
        "Peak GPU memory",
    )


def plot_kv_utilization_by_load(
    load_values: list[float],
    peak_utilization: list[list[float | None]],
    mean_utilization: list[list[float | None]],
    keeps_up: list[bool | None],
    out_path: str,
) -> None:
    """Peak utilization pins at 1.0 once any moment fills the pool; the mean shows how long
    the pool stays under pressure."""
    fig, ax = _figure("offered request rate (req/s)", "KV-cache utilization", "KV-cache pressure")
    _draw_spread(ax, load_values, peak_utilization, keeps_up, label="peak")
    _draw_spread(ax, load_values, mean_utilization, keeps_up, label="mean")
    ax.set_ylim(0, 1.05)
    ax.legend()
    _save(fig, out_path)


def plot_preemptions_by_load(
    load_values: list[float],
    preemptions: list[list[float | None]],
    keeps_up: list[bool | None],
    out_path: str,
) -> None:
    """Sequences evicted for KV blocks per run; each re-prefills everything it had."""
    _plot_rate_by_load(
        load_values, preemptions, keeps_up, out_path, "preemptions per run", "Preemptions"
    )


def plot_stage_latency_by_qps(runs: list[dict], out_path: str) -> None:
    """At each offered QPS, p50/p90/p99 of queue+prefill (up to the first token), of decode
    (first token to completion) and of the whole request, as side-by-side bars.

    `runs` is one `{"target_qps", "results"}` per QPS. Each bar is its own distribution's
    percentile, so the stage bars need not sum to the total bar: the slowest prefills and the
    slowest decodes are rarely the same requests."""
    fig, axes = _pyplot().subplots(1, 3, figsize=(12, 4), sharey=True)
    runs = sorted(runs, key=lambda r: r["target_qps"])
    stages = [_stage_latencies(run["results"]) for run in runs]
    x = range(len(runs))
    width = 0.27
    for ax, p in zip(axes, (50, 90, 99), strict=True):
        for i, label in enumerate(("queue + prefill", "decode", "total")):
            heights = [
                percentile(stage[i], p) * 1000 if len(stage[i]) >= min_samples(p) else float("nan")
                for stage in stages
            ]
            ax.bar([xi + (i - 1) * width for xi in x], heights, width, label=label)
        ax.set_xticks(list(x))
        ax.set_xticklabels([f"{run['target_qps']:g}" for run in runs])
        ax.set_xlabel("offered QPS")
        ax.set_title(f"p{p}")
    axes[0].set_ylabel("latency (ms)")
    axes[0].legend()
    _save(fig, out_path, keeps_up_note=False)


def _stage_latencies(results: list[dict]) -> tuple[list[float], list[float], list[float]]:
    """(queue+prefill, decode, total) durations of each successful request that reached a
    first token."""
    reached = [
        r for r in results if r.get("success", True) and r.get("first_token_latency") is not None
    ]
    return (
        [r["first_token_latency"] for r in reached],
        [r["latency"] - r["first_token_latency"] for r in reached],
        [r["latency"] for r in reached],
    )


def plot_latency_cdf_or_histogram(
    results: list[dict], out_path: str, kind: str = "cdf"
) -> None:
    """Per-request latency distribution for one run. `kind` is "cdf" or "hist"."""
    latencies = [r["latency"] * 1000 for r in results if r.get("success", True)]
    fig, ax = _figure("latency (ms)", "count" if kind == "hist" else "CDF", "Latency distribution")
    if kind == "hist":
        ax.hist(latencies, bins=50)
    else:
        ordered = sorted(latencies)
        ax.plot(ordered, [(i + 1) / len(ordered) for i in range(len(ordered))])
    _save(fig, out_path, keeps_up_note=False)


def plot_ablation_bar(
    labels: list[str], repeats: list[list[float]], out_path: str, ylabel: str
) -> None:
    """One bar per ablation arm at the median of its repeats, each repeat drawn as a dot."""
    fig, ax = _figure("", ylabel)
    ax.bar(labels, [median(r) for r in repeats], color="lightgray")
    for i, values in enumerate(repeats):
        ax.plot([i] * len(values), values, "o", color="black")
    _save(fig, out_path, keeps_up_note=False)


def plot_ablation_sweep(
    qps_by_arm: dict[str, list[float]],
    values_by_arm: dict[str, list[float | None]],
    keeps_up_by_arm: dict[str, list[bool | None]],
    out_path: str,
    ylabel: str,
) -> None:
    """One line per ablation arm against offered QPS, each arm over its own sweep. The QPS
    axis is logarithmic: arms can differ in capacity by an order of magnitude."""
    fig, ax = _figure("offered request rate (req/s)", ylabel)
    for arm, qps_values in qps_by_arm.items():
        _draw_series(ax, qps_values, values_by_arm[arm], keeps_up_by_arm[arm], label=arm)
    ax.set_xscale("log")
    ax.legend()
    _save(fig, out_path)
