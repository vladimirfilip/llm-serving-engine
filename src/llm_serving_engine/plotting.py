"""Benchmark plots.

Reads raw per-request result dicts (as written by loadgen/cli.py) and generates
plots from them. matplotlib is imported lazily inside each function so importing
this module never forces a matplotlib install.

Latency stats reuse llm_serving_engine.metrics.summarize/percentile; nothing here
recomputes percentile math.
"""

from __future__ import annotations

from .metrics import LatencySummary, percentile, summarize


def _successful_latencies(results: list[dict]) -> list[float]:
    return [r["latency"] for r in results if r.get("success", True)]


def _stage_latencies(results: list[dict]) -> tuple[list[float], list[float]]:
    """(queue+prefill, decode) durations per request that reached a first token:
    queue+prefill is client-measured first_token_latency; decode is what's left of the
    open-loop `latency` once that's subtracted.
    """
    reached_first_token = [
        r for r in results if r.get("success", True) and r.get("first_token_latency") is not None
    ]
    queue_and_prefill = [r["first_token_latency"] for r in reached_first_token]
    decode = [r["latency"] - r["first_token_latency"] for r in reached_first_token]
    return queue_and_prefill, decode


def plot_latency_pareto(results: list[dict], out_path: str) -> None:
    """Throughput-latency Pareto sweep.

    `results` is one run per QPS setting: [{"target_qps", "duration_s", "results": [...]}],
    where each inner "results" entry is a per-request dict with a "latency" key and an
    optional "success" flag. Achieved throughput is completed requests per second of
    wall-clock run time, not the offered target. Latency axis is milliseconds.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    throughputs, p50s, p95s, p99s = [], [], [], []
    for run in sorted(results, key=lambda r: r["target_qps"]):
        latencies = _successful_latencies(run["results"])
        if not latencies:
            continue
        throughputs.append(len(latencies) / run["duration_s"])
        summary = summarize(latencies)
        p50s.append(summary.p50 * 1000)
        p95s.append(summary.p95 * 1000)
        p99s.append(summary.p99 * 1000)

    fig, ax = plt.subplots()
    ax.plot(throughputs, p50s, marker="o", label="p50")
    ax.plot(throughputs, p95s, marker="o", label="p95")
    ax.plot(throughputs, p99s, marker="o", label="p99")
    ax.set_xlabel("achieved throughput (req/s)")
    ax.set_ylabel("latency (ms)")
    ax.set_title("Throughput-latency Pareto")
    ax.legend()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_percentiles_by_load(
    load_values: list[float],
    summaries: list[LatencySummary | None],
    out_path: str,
    ylabel: str,
    title: str,
    xlabel: str = "offered request rate (req/s)",
) -> None:
    """Shared renderer for the *_by_qps percentile plots: p50/p95/p99 (already in the
    unit `summaries` carries, converted to ms by the caller) against an offered-load
    axis. Points whose summary is None (e.g. a QPS point with no successful requests
    for that stage) are dropped rather than plotted as zero.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = [(load, s) for load, s in zip(load_values, summaries, strict=True) if s is not None]
    xs = [p[0] for p in points]

    fig, ax = plt.subplots()
    for pct, label in ((lambda s: s.p50, "p50"), (lambda s: s.p95, "p95"), (lambda s: s.p99, "p99")):
        ax.plot(xs, [pct(s) * 1000 for _x, s in points], marker="o", label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.savefig(out_path)
    plt.close(fig)


def plot_ttft_by_qps(
    qps_values: list[float], ttft_summaries: list[LatencySummary | None], out_path: str
) -> None:
    """Offered request rate vs p50/p95/p99 time-to-first-token."""
    _plot_percentiles_by_load(qps_values, ttft_summaries, out_path, "TTFT (ms)", "TTFT vs offered rate")


def plot_tpot_by_qps(
    qps_values: list[float], tpot_summaries: list[LatencySummary | None], out_path: str
) -> None:
    """Offered request rate vs p50/p95/p99 time-per-output-token."""
    _plot_percentiles_by_load(qps_values, tpot_summaries, out_path, "TPOT (ms)", "TPOT vs offered rate")


def plot_e2e_latency_by_qps(
    qps_values: list[float], e2e_summaries: list[LatencySummary | None], out_path: str
) -> None:
    """Offered request rate vs p50/p95/p99 end-to-end latency."""
    _plot_percentiles_by_load(
        qps_values, e2e_summaries, out_path, "end-to-end latency (ms)", "E2E latency vs offered rate"
    )


def _plot_scalar_by_load(
    load_values: list[float],
    values: list[float | None],
    out_path: str,
    ylabel: str,
    title: str,
    xlabel: str = "offered request rate (req/s)",
) -> None:
    """Shared renderer for single-line load-vs-metric plots (throughput, goodput, GPU
    memory, KV utilization). Points with a None value are dropped."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = [(load, v) for load, v in zip(load_values, values, strict=True) if v is not None]
    fig, ax = plt.subplots()
    ax.plot([p[0] for p in points], [p[1] for p in points], marker="o")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.savefig(out_path)
    plt.close(fig)


def plot_throughput_by_qps(qps_values: list[float], throughputs: list[float], out_path: str) -> None:
    """Offered request rate vs achieved request throughput."""
    _plot_scalar_by_load(
        qps_values, throughputs, out_path, "achieved throughput (req/s)", "Throughput vs offered rate"
    )


def plot_output_throughput_by_qps(
    qps_values: list[float], output_tokens_s: list[float], out_path: str
) -> None:
    """Offered request rate vs output-token throughput."""
    _plot_scalar_by_load(
        qps_values, output_tokens_s, out_path,
        "output-token throughput (tok/s)", "Output-token throughput vs offered rate",
    )


def plot_goodput_by_qps(qps_values: list[float], goodputs: list[float | None], out_path: str) -> None:
    """Offered request rate vs goodput (requests/s meeting every configured SLO)."""
    _plot_scalar_by_load(qps_values, goodputs, out_path, "goodput (req/s)", "Goodput vs offered rate")


def plot_gpu_memory_by_load(
    load_values: list[float], peak_memory_mb: list[float | None], out_path: str
) -> None:
    """Offered load vs peak GPU memory usage."""
    _plot_scalar_by_load(
        load_values, peak_memory_mb, out_path, "peak GPU memory (MB)",
        "Peak GPU memory vs offered load", xlabel="offered load (req/s)",
    )


def plot_kv_utilization_by_load(
    load_values: list[float], peak_kv_utilization: list[float | None], out_path: str
) -> None:
    """Offered load vs peak KV-cache utilization."""
    _plot_scalar_by_load(
        load_values, peak_kv_utilization, out_path, "peak KV-cache utilization",
        "Peak KV-cache utilization vs offered load", xlabel="offered load (req/s)",
    )


def plot_stage_latency_by_qps(results: list[dict], out_path: str) -> None:
    """p50/p90/p99 latency at each offered QPS, split into queue+prefill (up to the
    first token) and decode (first token to completion).

    `results` is the same one-run-per-QPS shape `plot_latency_pareto` takes. Each
    stage's percentile is computed independently over its own distribution across
    requests — an illustrative split of where time typically goes at that percentile,
    not a decomposition of any single request's own latency.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = sorted(results, key=lambda r: r["target_qps"])
    qps_labels = [f"{run['target_qps']:g}" for run in runs]
    stage_latencies = [_stage_latencies(run["results"]) for run in runs]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, p in zip(axes, (50, 90, 99), strict=True):
        queue = [percentile(q, p) * 1000 if q else 0.0 for q, _d in stage_latencies]
        decode = [percentile(d, p) * 1000 if d else 0.0 for _q, d in stage_latencies]
        x = range(len(qps_labels))
        ax.bar(x, queue, label="queue + prefill")
        ax.bar(x, decode, bottom=queue, label="decode")
        ax.set_xticks(list(x))
        ax.set_xticklabels(qps_labels)
        ax.set_xlabel("offered QPS")
        ax.set_title(f"p{p}")
    axes[0].set_ylabel("latency (ms)")
    axes[0].legend()
    fig.savefig(out_path)
    plt.close(fig)


def plot_latency_cdf_or_histogram(
    results: list[dict], out_path: str, kind: str = "cdf"
) -> None:
    """Per-request latency distribution for one run. `kind` is "cdf" or "hist"."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    latencies = [latency * 1000 for latency in _successful_latencies(results)]
    fig, ax = plt.subplots()
    if kind == "hist":
        ax.hist(latencies, bins=50)
        ax.set_ylabel("count")
    else:
        ordered = sorted(latencies)
        ys = [(i + 1) / len(ordered) for i in range(len(ordered))]
        ax.plot(ordered, ys)
        ax.set_ylabel("CDF")
    ax.set_xlabel("latency (ms)")
    ax.set_title("Latency distribution")
    fig.savefig(out_path)
    plt.close(fig)


def plot_ablation_bar(labels: list[str], values: list[float], out_path: str, ylabel: str) -> None:
    """Generic labeled bar chart for an ablation (e.g. static vs. continuous batching,
    contiguous vs. paged KV, fp16 vs. int8, with/without custom kernels): one bar per
    condition, comparing a single scalar such as throughput, p99, or decode-step time.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.bar(labels, values)
    ax.set_ylabel(ylabel)
    fig.savefig(out_path)
    plt.close(fig)
