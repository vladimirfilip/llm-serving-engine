import sqlite3

import pandas as pd
import pytest

from bench.run import Run
from bench.suites import nsys
from bench.suites.common import SuiteSkipped

MS = 1_000_000  # ns per ms
US = 1_000


def test_merging_takes_the_union_across_overlapping_streams():
    assert nsys.merge_intervals([(10, 20), (0, 5), (4, 8), (18, 30), (40, 41)]) == [
        (0, 8), (10, 30), (40, 41)]
    assert nsys.merge_intervals([]) == []


def test_busy_fraction_and_gap_statistics_follow_the_merged_union():
    # busy 0-1000, 1010-2000 (10 ns gap), 2100-3000 (100 ns gap) in a 3000 ns window
    intervals = [(0, 600), (400, 1000), (1010, 2000), (2100, 3000)]
    out = nsys.busy_and_gaps(intervals, (0, 3000))
    assert out["gpu_busy_fraction"] == pytest.approx((1000 + 990 + 900) / 3000)
    assert sorted(out["gaps_s"] * 1e9) == pytest.approx([10, 100])
    assert out["gap_p50"] == pytest.approx(55e-9)
    assert out["gap_time_fraction_over_50us"] == 0


def test_only_gaps_longer_than_fifty_microseconds_count_toward_the_idle_fraction():
    intervals = [(0, 100 * US), (200 * US, 300 * US), (310 * US, 400 * US)]
    out = nsys.busy_and_gaps(intervals, (0, 400 * US))
    assert out["gap_time_fraction_over_50us"] == pytest.approx(100 * US / (400 * US))


def make_db(path, ranges, kernels, inline_text=True):
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER)")
    db.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (start INTEGER, end INTEGER)")
    db.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?)", kernels)
    if inline_text:
        db.execute("CREATE TABLE NVTX_EVENTS (text TEXT, start INTEGER, end INTEGER)")
        db.executemany("INSERT INTO NVTX_EVENTS VALUES (?, ?, ?)", ranges)
    else:
        db.execute("CREATE TABLE StringIds (id INTEGER, value TEXT)")
        db.execute("CREATE TABLE NVTX_EVENTS "
                   "(text TEXT, textId INTEGER, start INTEGER, end INTEGER)")
        names = sorted({r[0] for r in ranges})
        db.executemany("INSERT INTO StringIds VALUES (?, ?)", list(enumerate(names)))
        db.executemany("INSERT INTO NVTX_EVENTS VALUES (NULL, ?, ?, ?)",
                       [(names.index(n), s, e) for n, s, e in ranges])
    db.commit()
    return db


RANGES = [
    ("step", 0, 10 * MS), ("schedule", 0, 1 * MS), ("prepare_inputs", 2 * MS, 3 * MS),
    ("forward", 3 * MS, 8 * MS), ("sample", 8 * MS, 9 * MS), ("postprocess", 9 * MS, 10 * MS),
    ("step", 10 * MS, 20 * MS), ("forward", 12 * MS, 18 * MS), ("step", 20 * MS, 30 * MS),
]
KERNELS = [(3 * MS, 8 * MS), (12 * MS, 18 * MS)]


@pytest.mark.parametrize("inline_text", [True, False])
def test_step_breakdown_reports_each_ranges_time_and_gpu_idle_inside_the_period(
    tmp_path, inline_text
):
    db = make_db(tmp_path / "x.sqlite", RANGES, KERNELS, inline_text)
    steps = nsys.step_breakdown(nsys.read_nvtx(db), nsys.read_device_intervals(db))
    assert len(steps) == 2  # the last step has no successor to give it a period
    first = steps.iloc[0]
    assert first.period_s == pytest.approx(0.010)
    assert first.forward_s == pytest.approx(0.005) and first.schedule_s == pytest.approx(0.001)
    assert first.gpu_busy_s == pytest.approx(0.005) and first.gpu_idle_s == pytest.approx(0.005)
    assert steps.iloc[1].gpu_busy_s == pytest.approx(0.006)
    assert steps.iloc[1].gpu_idle_s == pytest.approx(0.004)


def test_analyse_summarises_a_capture_and_breaks_down_only_ours_steps(tmp_path):
    make_db(tmp_path / "ours.sqlite", RANGES, KERNELS)
    row, gaps, steps = nsys.analyse(tmp_path / "ours.sqlite", "ours", 16, capture_s=0.020)
    assert row["engine"] == "ours" and row["batch"] == 16
    # 11 ms busy in a 20 ms capture that starts with the first kernel at 3 ms
    assert row["gpu_busy_fraction"] == pytest.approx(11 / 20)
    assert row["window_source"].startswith("capture_s from the first device event")
    assert len(gaps) == 1 and gaps.gap_s.iloc[0] == pytest.approx(0.004)
    assert isinstance(steps, pd.DataFrame) and len(steps) == 2 and (steps.batch == 16).all()
    _, _, others = nsys.analyse(tmp_path / "ours.sqlite", "vllm", 16, capture_s=0.020)
    assert others.empty


def test_the_capture_window_comes_from_nsight_when_the_export_records_it(tmp_path):
    db = make_db(tmp_path / "x.sqlite", RANGES, KERNELS)
    db.execute("CREATE TABLE ANALYSIS_DETAILS (startTime INTEGER, stopTime INTEGER)")
    db.execute("INSERT INTO ANALYSIS_DETAILS VALUES (0, ?)", (30 * MS,))
    db.commit()
    row, _, _ = nsys.analyse(tmp_path / "x.sqlite", "ours", 1, capture_s=0.020)
    assert row["window_source"] == "nsight capture bounds"
    assert row["gpu_busy_fraction"] == pytest.approx(11 / 30)


def test_an_export_without_nvtx_still_gives_gpu_statistics(tmp_path):
    db = sqlite3.connect(tmp_path / "plain.sqlite")
    db.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER)")
    db.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?)", KERNELS)
    db.commit()
    row, gaps, steps = nsys.analyse(tmp_path / "plain.sqlite", "ours", 1, capture_s=0.020)
    assert row["gpu_busy_fraction"] == pytest.approx(11 / 20) and len(gaps) == 1
    assert steps.empty


def test_the_suite_is_skipped_when_nsight_is_not_installed(fast_config, tmp_path):
    run = Run.open(fast_config, "n", results_dir=tmp_path)
    run.cfg.hardware["nsys_path"] = "definitely-not-nsys"
    with pytest.raises(SuiteSkipped, match="not on the path"):
        nsys.execute(run, ["ours"])


def test_activity_past_the_capture_window_cannot_push_the_busy_fraction_over_one():
    out = nsys.busy_and_gaps([(0, 10_400 * MS)], (0, 10_000 * MS))
    assert out["gpu_busy_fraction"] == pytest.approx(1.0)
    clipped = nsys.busy_and_gaps([(0, 4 * MS), (6 * MS, 30 * MS)], (0, 10 * MS))
    assert clipped["gpu_busy_fraction"] == pytest.approx(0.8)
    assert sorted(clipped["gaps_s"] * 1e9) == pytest.approx([2 * MS])


def test_recorded_bounds_that_span_the_whole_session_are_rejected_with_the_reason(tmp_path):
    db = make_db(tmp_path / "x.sqlite", RANGES, KERNELS)
    db.execute("CREATE TABLE ANALYSIS_DETAILS (startTime INTEGER, stopTime INTEGER)")
    db.execute("INSERT INTO ANALYSIS_DETAILS VALUES (0, ?)", (200_000 * MS,))  # a 200 s session
    db.commit()
    row, _, _ = nsys.analyse(tmp_path / "x.sqlite", "ours", 1, capture_s=0.020)
    assert "recorded bounds rejected" in row["window_source"]
    assert row["gpu_busy_fraction"] == pytest.approx(11 / 20)


def test_an_analysis_table_with_other_column_names_falls_back_instead_of_crashing(tmp_path):
    db = make_db(tmp_path / "x.sqlite", RANGES, KERNELS)
    db.execute("CREATE TABLE ANALYSIS_DETAILS (begin INTEGER, finish INTEGER)")
    db.commit()
    row, _, _ = nsys.analyse(tmp_path / "x.sqlite", "ours", 1, capture_s=0.020)
    assert row["window_source"].startswith("capture_s from the first device event")


def test_a_capture_with_no_device_events_is_reported_not_a_crash(tmp_path):
    db = sqlite3.connect(tmp_path / "empty.sqlite")
    db.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER)")
    db.commit()
    row, gaps, steps = nsys.analyse(tmp_path / "empty.sqlite", "ours", 1, capture_s=0.020)
    assert row["window_source"] == "no device events in the capture"
    assert row["gpu_busy_fraction"] is None and gaps.empty and steps.empty
