import random

import pytest

from llm_serving_engine.loadgen.report import summarize_latencies
from llm_serving_engine.observability.metrics import summarize
from llm_serving_engine.observability.plotting import (
    plot_ablation_bar,
    plot_ablation_sweep,
    plot_e2e_latency_by_qps,
    plot_goodput_by_qps,
    plot_gpu_memory_by_load,
    plot_itl_by_qps,
    plot_kv_utilization_by_load,
    plot_latency_cdf_or_histogram,
    plot_latency_pareto,
    plot_output_throughput_by_qps,
    plot_preemptions_by_load,
    plot_stage_latency_by_qps,
    plot_throughput_by_qps,
    plot_tpot_by_qps,
    plot_ttft_by_qps,
    trusted_ms,
)

QPS = [1, 2, 4]
KEEPS_UP = [True, False, None]


def _requests(n: int, base: float, shape: str = "chat") -> list[dict]:
    rng = random.Random(0)
    results = []
    for i in range(n):
        ttft = base + rng.random() * 0.01
        results.append({
            "scheduled_at": float(i), "shape": shape, "success": True,
            "first_token_latency": ttft, "latency": ttft + 0.05 + rng.random() * 0.01,
            "completed_at": i + 0.2, "token_times": [0.0, 0.02, 0.05],
        })
    return results


def _summaries(n: int, base: float, count: int = 120) -> list:
    rng = random.Random(0)
    return [summarize([base + rng.random() * 0.01 for _ in range(count)]) for _ in range(n)]


def _written(path) -> bool:
    return path.exists() and path.stat().st_size > 0


def test_trusted_ms_drops_a_percentile_that_is_only_the_maximum():
    assert trusted_ms(summarize([0.001] * 99 + [1.0]), "p99") == pytest.approx(1.0)
    assert trusted_ms(summarize([0.001] * 98 + [1.0]), "p99") is None
    assert trusted_ms(summarize([0.001] * 98 + [1.0]), "p50") == pytest.approx(1.0)
    assert trusted_ms(None, "p50") is None


def test_plot_latency_pareto_draws_each_shape(tmp_path):
    out = tmp_path / "pareto.png"
    latencies = [
        summarize_latencies(_requests(150, 0.01 * q) + _requests(5, 0.5 * q, "document"))
        for q in QPS
    ]
    plot_latency_pareto([[0.9, 1.0], [1.8, 2.0], [3.1, 3.3]], latencies, KEEPS_UP, str(out))
    assert _written(out)


def test_plot_latency_cdf_and_histogram_write_files(tmp_path):
    for kind in ("cdf", "hist"):
        out = tmp_path / f"{kind}.png"
        plot_latency_cdf_or_histogram(_requests(100, 0.1), str(out), kind=kind)
        assert _written(out)


def test_plot_stage_latency_by_qps_skips_percentiles_with_too_few_requests(tmp_path):
    out = tmp_path / "stage_latency.png"
    runs = [
        {"target_qps": q, "results": _requests(n, 0.01 * q)}
        for q, n in zip(QPS, (150, 30, 1), strict=True)
    ]
    plot_stage_latency_by_qps(runs, str(out))
    assert _written(out)


def test_plot_ablation_bar_draws_every_repeat(tmp_path):
    out = tmp_path / "ablation.png"
    plot_ablation_bar(["static", "continuous"], [[1.2, 1.3, 1.1], [3.4]], str(out), "req/s")
    assert _written(out)


def test_plot_ablation_sweep_draws_arms_over_their_own_ranges(tmp_path):
    out = tmp_path / "sweep.png"
    plot_ablation_sweep(
        {"true": [0.7, 1.4, 2.0], "false": [0.03, 0.05], "broken": []},
        {"true": [100.0, 120.0, 180.0], "false": [150.0, None], "broken": []},
        {"true": [True, True, False], "false": [None, False], "broken": []},
        str(out),
        ylabel="e2e p99 (ms)",
    )
    assert _written(out)


def test_plot_ttft_by_qps_draws_each_shape_and_drops_its_missing_points(tmp_path):
    out = tmp_path / "ttft.png"
    chat, document = _summaries(3, 0.05), _summaries(3, 0.5, count=4)
    by_shape = [
        {"chat": chat[0], "document": document[0]},
        {"chat": chat[1]},  # no document completed at this point
        {"chat": chat[2], "document": document[2]},
    ]
    plot_ttft_by_qps(QPS, by_shape, KEEPS_UP, str(out))
    assert _written(out)


@pytest.mark.parametrize("plot", [plot_tpot_by_qps, plot_itl_by_qps, plot_e2e_latency_by_qps])
def test_percentile_plots_skip_points_without_a_summary(tmp_path, plot):
    out = tmp_path / "percentiles.png"
    summaries = _summaries(2, 0.05)
    plot(QPS, [summaries[0], None, summaries[1]], KEEPS_UP, str(out))
    assert _written(out)


def test_plot_throughput_by_qps_draws_arrivals_beside_throughput(tmp_path):
    out = tmp_path / "throughput.png"
    plot_throughput_by_qps(
        QPS, [[0.9, 1.0], [1.8, 1.9], [3.1, 3.3]], [[1.0, 1.1], [2.0, 1.9], [4.1, 3.9]],
        KEEPS_UP, str(out),
    )
    assert _written(out)


@pytest.mark.parametrize(
    "plot",
    [plot_output_throughput_by_qps, plot_goodput_by_qps, plot_gpu_memory_by_load,
     plot_preemptions_by_load],
)
def test_rate_plots_draw_repeat_spread_and_skip_missing_values(tmp_path, plot):
    out = tmp_path / "rate.png"
    plot(QPS, [[20.0, 22.0], [38.0, None], [None, None]], KEEPS_UP, str(out))
    assert _written(out)


def test_plot_kv_utilization_by_load_draws_peak_and_mean(tmp_path):
    out = tmp_path / "kv_utilization.png"
    plot_kv_utilization_by_load(
        QPS, [[0.3], [0.9], [1.0]], [[0.1], [0.4], [None]], KEEPS_UP, str(out)
    )
    assert _written(out)
