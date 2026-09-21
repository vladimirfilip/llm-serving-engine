import json

import pandas as pd
import pytest

from bench.run import Run
from bench.suites import probe, sweep, tune
from bench.suites.sweep import point_done, point_paths


@pytest.fixture
def run(fast_config, tmp_path, synthetic_datasets):
    return Run.open(fast_config, "sw", results_dir=tmp_path / "results")


def expected_points(run) -> set[tuple]:
    return {(w, round(r, 3), rep) for w, _i, r, rep in sweep.plan(run, ["mock"])}


def on_disk(run) -> set[tuple]:
    found = set()
    for path in (run.dir / "sweep" / "mock").glob("*/*.parquet"):
        rate, rep = path.stem.removeprefix("rate").split("_rep")
        found.add((path.parent.name, round(float(rate), 3), int(rep)))
    return found


def test_a_sweep_writes_one_valid_point_per_workload_rate_and_repeat(run):
    sweep.execute(run, ["mock"])
    assert on_disk(run) == expected_points(run)
    points = pd.read_csv(run.dir / "tables" / "sweep_points.csv")
    assert set(points.engine) == {"mock"} and points.valid.all()
    assert (points.out_tok_s > 0).all() and (points.error_rate == 0).all()
    assert {"ttft_p99", "tpot_p50", "itl_p99", "slo_attainment", "goodput_rps"} <= set(points)
    summary = pd.read_csv(run.dir / "tables" / "sweep_summary.csv")
    assert set(summary.workload) == {"sharegpt", "short_short", "long_short", "short_long"}


def test_a_killed_sweep_resumes_without_duplicate_or_missing_points(run, monkeypatch):
    real, done = sweep.run_point, []

    def dies_after_three(*args, **kwargs):
        if len(done) == 3:
            raise KeyboardInterrupt
        done.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(sweep, "run_point", dies_after_three)
    with pytest.raises(KeyboardInterrupt):
        sweep.execute(run, ["mock"])
    assert len(on_disk(run)) == 3
    first = {p: p.stat().st_mtime_ns for p in (run.dir / "sweep").rglob("*.parquet")}

    monkeypatch.setattr(sweep, "run_point", real)
    sweep.execute(run, ["mock"])
    assert on_disk(run) == expected_points(run)
    assert len(list((run.dir / "sweep").rglob("*.json"))) == len(expected_points(run))
    assert all(p.stat().st_mtime_ns == t for p, t in first.items())


def test_a_point_with_a_torn_parquet_is_redone_and_a_tmp_file_is_ignored(run):
    sweep.execute(run, ["mock"])
    workload, rate, rep = sorted(expected_points(run))[0]
    parquet, meta = point_paths(run, "mock", workload, rate, rep)
    assert point_done(parquet, meta)
    parquet.write_bytes(b"torn")
    parquet.with_suffix(".tmp").write_bytes(b"partial")
    assert not point_done(parquet, meta)
    sweep.execute(run, ["mock"])
    assert point_done(parquet, meta)


def test_points_are_shuffled_with_the_suite_seed_not_ascending(run):
    order = sweep.plan(run, ["mock"])
    assert order == sweep.plan(run, ["mock"])
    rates = [(w, r) for w, _i, r, _rep in order]
    assert rates != sorted(rates)


def test_capacity_probe_places_the_sweep_rates_from_the_best_engine(run):
    probe.execute(run, ["mock"])
    capacity = json.loads(probe.capacity_path(run).read_text())["mock"]
    assert set(capacity) == {"sharegpt", "short_short", "long_short", "short_long"}
    assert all(c["capacity_rps"] > 0 and c["capacity_tok_s"] > 0 for c in capacity.values())
    assert probe.ref_capacity_rps(run, "sharegpt", ["mock"]) == capacity["sharegpt"]["capacity_rps"]
    with pytest.raises(RuntimeError, match="no capacity probe for vllm"):
        probe.ref_capacity_rps(run, "sharegpt", ["mock", "vllm"])


def test_tuning_picks_the_highest_throughput_and_the_smaller_budget_within_two_percent():
    assert tune.pick_budget({2048: 100.0, 4096: 130.0, 8192: 90.0}) == 4096
    assert tune.pick_budget({2048: 128.0, 4096: 130.0, 8192: 100.0}) == 2048
    assert tune.pick_budget({2048: 100.0}) == 2048


def test_an_engine_with_no_tuning_knob_keeps_a_fixed_launch(run):
    tune.execute(run, ["mock"])
    assert json.loads((run.dir / "tune" / "tuned.json").read_text())["mock"]["token_budget"] is None
