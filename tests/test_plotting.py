import random

from llm_serving_engine.observability.metrics import summarize
from llm_serving_engine.observability.plotting import (
    plot_ablation_bar,
    plot_ablation_sweep,
    plot_e2e_latency_by_qps,
    plot_goodput_by_qps,
    plot_gpu_memory_by_load,
    plot_kv_utilization_by_load,
    plot_latency_cdf_or_histogram,
    plot_latency_pareto,
    plot_output_throughput_by_qps,
    plot_stage_latency_by_qps,
    plot_throughput_by_qps,
    plot_tpot_by_qps,
    plot_ttft_by_qps,
)


def _synthetic_requests(n: int, base_latency: float) -> list[dict]:
    rng = random.Random(0)
    return [
        {"latency": base_latency + rng.random() * 0.05, "completed_at": i * 0.2, "success": True}
        for i in range(n)
    ]


def _synthetic_staged_requests(n: int, queue_and_prefill: float, decode: float) -> list[dict]:
    rng = random.Random(0)
    results = []
    for _ in range(n):
        ftl = queue_and_prefill + rng.random() * 0.01
        results.append({
            "success": True,
            "first_token_latency": ftl,
            "latency": ftl + decode + rng.random() * 0.01,
        })
    return results


def test_plot_latency_pareto_writes_nonempty_file(tmp_path):
    out = tmp_path / "pareto.png"
    runs = [
        {"target_qps": qps, "duration_s": 10.0, "results": _synthetic_requests(50, 0.01 * qps)}
        for qps in (1, 5, 10)
    ]
    plot_latency_pareto(runs, str(out))
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_latency_cdf_writes_nonempty_file(tmp_path):
    out = tmp_path / "cdf.png"
    plot_latency_cdf_or_histogram(_synthetic_requests(100, 0.1), str(out))
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_latency_histogram_writes_nonempty_file(tmp_path):
    out = tmp_path / "hist.png"
    plot_latency_cdf_or_histogram(_synthetic_requests(100, 0.1), str(out), kind="hist")
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_stage_latency_by_qps_writes_nonempty_file(tmp_path):
    out = tmp_path / "stage_latency.png"
    runs = [
        {
            "target_qps": qps,
            "duration_s": 10.0,
            "results": _synthetic_staged_requests(50, queue_and_prefill=0.01 * qps, decode=0.05),
        }
        for qps in (1, 5, 10)
    ]
    plot_stage_latency_by_qps(runs, str(out))
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_ablation_bar_writes_nonempty_file(tmp_path):
    out = tmp_path / "ablation.png"
    plot_ablation_bar(["static", "continuous"], [120.0, 340.0], str(out), ylabel="capacity (req/s)")
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_ablation_sweep_draws_every_arm_and_drops_none_values(tmp_path):
    out = tmp_path / "sweep.png"
    plot_ablation_sweep(
        [1.0, 2.0, 4.0],
        {"paged": [100.0, 120.0, 180.0], "contiguous": [150.0, None, 9000.0]},
        {"paged": [True, True, True], "contiguous": [True, False, False]},
        str(out),
        ylabel="e2e p99 (ms)",
    )
    assert out.stat().st_size > 0


def _synthetic_summaries(n: int, base: float) -> list:
    rng = random.Random(0)
    return [summarize([base + rng.random() * 0.01 for _ in range(20)]) for _ in range(n)]


def test_plot_ttft_by_qps_draws_each_shape_and_drops_its_missing_points(tmp_path):
    out = tmp_path / "ttft.png"
    chat, document = _synthetic_summaries(3, 0.05), _synthetic_summaries(3, 0.5)
    by_shape = [
        {"chat": chat[0], "document": document[0]},
        {"chat": chat[1]},  # no document completed at this point
        {"chat": chat[2], "document": document[2]},
    ]
    plot_ttft_by_qps([1, 2, 4], by_shape, str(out))
    assert out.exists() and out.stat().st_size > 0


def test_plot_tpot_by_qps_writes_nonempty_file(tmp_path):
    out = tmp_path / "tpot.png"
    plot_tpot_by_qps([1, 2, 4], _synthetic_summaries(3, 0.02), str(out))
    assert out.exists() and out.stat().st_size > 0


def test_plot_e2e_latency_by_qps_writes_nonempty_file(tmp_path):
    out = tmp_path / "e2e.png"
    plot_e2e_latency_by_qps([1, 2, 4], _synthetic_summaries(3, 0.5), str(out))
    assert out.exists() and out.stat().st_size > 0


def test_plot_percentile_by_qps_skips_none_points(tmp_path):
    out = tmp_path / "tpot_partial.png"
    summaries = _synthetic_summaries(2, 0.05)
    plot_tpot_by_qps([1, 2, 4], [summaries[0], None, summaries[1]], str(out))
    assert out.exists() and out.stat().st_size > 0


def test_plot_throughput_by_qps_writes_nonempty_file(tmp_path):
    out = tmp_path / "throughput.png"
    plot_throughput_by_qps([1, 2, 4], [0.9, 1.8, 3.5], str(out))
    assert out.exists() and out.stat().st_size > 0


def test_plot_output_throughput_by_qps_writes_nonempty_file(tmp_path):
    out = tmp_path / "output_throughput.png"
    plot_output_throughput_by_qps([1, 2, 4], [20.0, 38.0, 70.0], str(out))
    assert out.exists() and out.stat().st_size > 0


def test_plot_goodput_by_qps_writes_nonempty_file(tmp_path):
    out = tmp_path / "goodput.png"
    plot_goodput_by_qps([1, 2, 4], [0.9, 1.5, None], str(out))
    assert out.exists() and out.stat().st_size > 0


def test_plot_gpu_memory_by_load_writes_nonempty_file(tmp_path):
    out = tmp_path / "gpu_memory.png"
    plot_gpu_memory_by_load([1, 2, 4], [1000.0, 1200.0, None], str(out))
    assert out.exists() and out.stat().st_size > 0


def test_plot_kv_utilization_by_load_writes_nonempty_file(tmp_path):
    out = tmp_path / "kv_utilization.png"
    plot_kv_utilization_by_load([1, 2, 4], [0.3, 0.6, None], str(out))
    assert out.exists() and out.stat().st_size > 0
