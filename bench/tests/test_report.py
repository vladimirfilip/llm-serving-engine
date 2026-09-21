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
