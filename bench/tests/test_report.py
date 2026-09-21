import pytest

from bench.plots import PLOTS
from bench.report import build_report, md_table
from bench.tests import fixture_run


@pytest.fixture(scope="module")
def full_run(tmp_path_factory):
    return fixture_run.write(tmp_path_factory.mktemp("results") / "r-full", fixture_run.ALL_SUITES)


def test_every_plot_is_drawn_when_its_suite_has_data(full_run):
    _, info = build_report(full_run)
    assert set(info["drawn"]) == set(PLOTS) and all(info["drawn"].values()), {
        k: v for k, v in info["drawn"].items() if not v}
    for key, (name, _title) in PLOTS.items():
        for suffix in ("png", "svg"):
            assert (full_run / "plots" / f"{name}.{suffix}").stat().st_size > 1000, key


def test_the_report_lists_sections_in_order_with_images_and_a_headline_table(full_run):
    text = build_report(full_run)[0].read_text()
    order = ["## Setup", "## Headline table", "## Headline plots", "## Correctness",
             "## Load results", "## Single-stream", "## Kernels", "## System overhead",
             "## Memory", "## Scheduler", "## Ablation", "## Energy, cold start, soak",
             "## Method notes"]
    assert [text.index(h) for h in order] == sorted(text.index(h) for h in order)
    assert "![" in text and "plots/p01_throughput_latency_sharegpt.png" in text
    headline = text.split("## Headline table")[1].split("##")[0]
    assert "| vllm |" in headline and "| ours |" in headline
    assert "Overload behaviour" in text and "flash_attn: flash-attn is not installed" in text


def test_headline_numbers_are_read_from_the_tables(full_run):
    _, info = build_report(full_run)
    row = info["headline"].set_index("engine").loc["ours"]
    assert row.max_sustainable_rps == 2.0 and row.capacity_tok_s == 700.0
    assert row.b1_tpot_ms == pytest.approx((0.012 * 1.3 + 512 * 2e-7) * 1000)
    assert row.correctness == "pass"
    assert row.cold_start_s == 70.0 and row.first_request_penalty_ms == pytest.approx(50.0)


def test_suites_that_did_not_run_are_listed_not_run_and_nothing_else_is_drawn(tmp_path):
    run = fixture_run.write(tmp_path / "r-partial", {"probe", "sweep"})
    report, info = build_report(run)
    drawn = {k for k, v in info["drawn"].items() if v}
    assert drawn == {"p01", "p06", "p07", "p08", "p09", "p23"}
    text = report.read_text()
    assert "*ablation of ours' optimizations: not run.*" in text
    assert "*batch-1 decode against the bandwidth bound: not run.*" in text
    assert "*long-run stability: not run.*" in text


def test_banners_report_failed_gates_unlocked_clocks_quick_runs_and_invalid_points(tmp_path):
    run = fixture_run.write(tmp_path / "r-bad", {"probe", "sweep", "correctness"}, quick=True,
                            locked=False, invalid=True, gates_ok=False)
    text = build_report(run)[0].read_text()
    notices = text.split("## Notices")[1].split("## Setup")[0]
    assert "ENGINE FAILED CORRECTNESS" in notices and "Clocks were not locked" in notices
    assert "Quick run" in notices and "stayed invalid or throttled" in notices


def test_a_report_with_no_tables_still_builds_and_marks_everything_not_run(tmp_path):
    run = fixture_run.write(tmp_path / "r-empty", set())
    _, info = build_report(run)
    assert not any(info["drawn"].values())


def test_md_table_formats_numbers_and_marks_missing_values():
    import pandas as pd

    table = md_table(pd.DataFrame({"engine": ["a", "b"], "x": [1.23456, None]}), {"x": ".2f"})
    assert table.splitlines() == ["| engine | x |", "|---|---|", "| a | 1.23 |", "| b | n/a |"]


def test_the_headline_takes_the_median_of_repeats_and_ignores_invalid_points(tmp_path):
    run = fixture_run.write(tmp_path / "r", {"probe", "sweep"}, invalid=True)
    _, info = build_report(run)
    row = info["headline"].set_index("engine")
    # repeats at rate 2.0 hold 79.6 and 83.6 ms: the median, the value P01 plots
    assert row.loc["vllm", "p99_tpot_ms_at_half_ref"] == pytest.approx(81.59, abs=0.01)
    # ours has repeat 1 invalid at that rate: only the valid repeat (79.6 ms) is used
    assert row.loc["ours", "p99_tpot_ms_at_half_ref"] == pytest.approx(79.6, abs=0.01)


def test_the_setup_block_names_engine_versions_and_each_resolved_launch_command(full_run):
    text = build_report(full_run)[0].read_text()
    setup = text.split("## Setup")[1].split("## Headline table")[0]
    assert "baselines vllm 0.29.0" in setup and "ours at commit abc1234" in setup
    assert "Launch, ours (token budget 4096): `" in setup and "bench.engines.ours_server" in setup


def test_missing_stats_dropped_contenders_and_skipped_cells_appear_in_the_report(tmp_path):
    import json

    run = fixture_run.write(tmp_path / "r", fixture_run.ALL_SUITES)
    checks = json.loads((run / "checks.json").read_text())
    checks["vllm"]["stats_available"] = False
    (run / "checks.json").write_text(json.dumps(checks))
    text = build_report(run)[0].read_text()
    assert "vllm: no stats endpoint; queue, KV and memory plots show n/a." in text
    assert "No stats endpoint for vllm" in text
    assert "dropped for numeric error above tolerance: flashinfer" in text
    assert "batch 128 x context 8192" in text


def test_the_headline_correctness_cell_is_not_evaluated_when_any_gate_is_unevaluated(tmp_path):
    import json

    run = fixture_run.write(tmp_path / "r", {"probe", "sweep", "correctness"})
    summary = json.loads((run / "correctness" / "summary.json").read_text())
    summary["gates"]["gsm8k_vs_vllm"]["passed"] = None
    (run / "correctness" / "summary.json").write_text(json.dumps(summary))
    assert build_report(run)[1]["headline"].set_index("engine").loc["ours", "correctness"] == \
        "not evaluated"


def test_the_memory_section_carries_the_breakdown_plot(full_run):
    text = build_report(full_run)[0].read_text()
    assert "p04_memory_breakdown.png" in text.split("## Memory")[1].split("## Scheduler")[0]


def test_the_decode_bandwidth_twin_axis_maps_the_same_range_as_the_left_axis(full_run):
    import matplotlib.pyplot as plt

    from bench.plots import p14_attn_decode
    from bench.plots.style import PlotContext

    ctx = PlotContext(full_run, full_run / "plots", fixture_run.env_json(), {})
    captured = {}
    original = plt.close

    def keep(fig):
        captured["fig"] = fig

    plt.close = keep
    try:
        p14_attn_decode.draw(ctx)
    finally:
        plt.close = original
    ax, twin = captured["fig"].axes[0], captured["fig"].axes[3]
    low, high = ax.get_ylim()
    assert tuple(twin.get_ylim()) == pytest.approx((low / 637 * 100, high / 637 * 100))


def test_the_interference_plot_is_sized_by_its_panels_not_its_combinations(tmp_path):
    import matplotlib.pyplot as plt

    from bench.plots import p21_prefill_interference
    from bench.plots.style import PlotContext

    run = fixture_run.write(tmp_path / "r", {"scheduler"})
    source = run / "scheduler" / "ours" / "interference_default.parquet"
    for engine in ("vllm", "sglang", "trtllm"):
        (run / "scheduler" / engine).mkdir(exist_ok=True)
        for variant in ("default", "chunked_prefill_off"):
            (run / "scheduler" / engine / f"interference_{variant}.parquet").write_bytes(
                source.read_bytes())
    ctx = PlotContext(run, run / "plots", fixture_run.env_json(), {})
    captured = {}
    original = plt.close
    plt.close = lambda fig: captured.setdefault("fig", fig)
    try:
        p21_prefill_interference.draw(ctx)
    finally:
        plt.close = original
    width, height = captured["fig"].get_size_inches()
    assert len(captured["fig"].axes) == 2 * 7 and (width, height) == (7 * 7, 4.5 * 8)


def test_a_run_whose_every_profile_was_empty_still_builds_a_report(tmp_path):
    import pandas as pd

    run = fixture_run.write(tmp_path / "r", {"sweep"})
    pd.DataFrame([{"engine": "ours", "batch": 16, "gpu_busy_fraction": None, "gap_p50": None,
                   "gap_p99": None, "gap_time_fraction_over_50us": None,
                   "window_source": "no device events in the capture"}]
                 ).to_csv(run / "tables" / "nsys.csv", index=False)
    (run / "tables" / "nsys_steps.csv").write_text(pd.DataFrame([]).to_csv(index=False))
    _, info = build_report(run)
    assert info["drawn"]["p17"] is None and info["drawn"]["p18"] is None


def test_a_table_a_suite_left_with_no_columns_does_not_stop_the_report(tmp_path):
    import pandas as pd

    run = fixture_run.write(tmp_path / "r", {"probe", "sweep", "scheduler"})
    for name in ("scheduler_interference", "scheduler_overload", "soak", "kernels_decode"):
        (run / "tables" / f"{name}.csv").write_text(pd.DataFrame([]).to_csv(index=False))
    _, info = build_report(run)
    assert info["drawn"]["p01"] and info["drawn"]["p22"] is not None
