import pandas as pd
import pytest
import yaml

from bench.config import load_config
from bench.run import Run
from bench.suites import memory, probe
from bench.suites.common import engine_session, load_datasets


def rec(status="ok", last_token=None):
    return {"status": status, "token_times": [] if last_token is None else [last_token]}


def test_exhaustion_outcomes_are_told_apart_by_crash_completion_and_silence():
    ok, timeout = rec(), rec("timeout", last_token=10.0)
    crashed = memory.classify_exhaustion([ok, ok], 5, server_exited=True, hang_s=120)
    assert crashed == "server_crashed"
    assert memory.classify_exhaustion([ok, ok], 5, False, 120) == "completed_all"
    # no token from any stream for 290 s while requests are unfinished
    assert memory.classify_exhaustion([ok, timeout], 300, False, 120) == "hang"
    # requests failed, but tokens were still arriving 20 s before the end
    assert memory.classify_exhaustion([rec("error", 280.0), timeout], 300, False, 120) == \
        "partial_errors"
    assert memory.classify_exhaustion([rec("error")], 3, False, 120) == "partial_errors"


def test_preemptions_and_peak_running_come_from_the_stats_samples():
    samples = [{"preemptions_total": 4, "running": 2}, {"preemptions_total": 9, "running": 7},
               {"preemptions_total": 9, "running": 5}]
    assert memory.preemptions(samples) == 5 and memory.peak_running(samples) == 7
    assert memory.preemptions([{"running": None}]) is None and memory.peak_running([]) is None


def test_kv_utilisation_and_internal_fragmentation_follow_their_definitions():
    kv = {"block_size": 16, "blocks_total": 100, "blocks_used": 10, "tokens_used": 100}
    series = memory.kv_series([{"t": 11.0, "kv": kv}], t0=10.0)
    assert series.kv_util[0] == pytest.approx(0.10)
    assert series.internal_frag[0] == pytest.approx(1 - 100 / 160)
    baseline = memory.kv_series([{"t": 2.0, "kv": {"usage": 0.4}}], t0=1.0)
    assert baseline.kv_util[0] == 0.4 and baseline.internal_frag[0] is None


@pytest.fixture
def run(fast_config, tmp_path, synthetic_datasets):
    return Run.open(fast_config, "mem", results_dir=tmp_path / "results")


def test_the_grid_stops_a_row_at_its_first_failure_relaunches_and_resumes(
    fast_config_dir, tmp_path, synthetic_datasets
):
    suite = yaml.safe_load((fast_config_dir / "suite.yaml").read_text())
    suite["memory"]["grid_batches"] = [1, 2, 3]
    (fast_config_dir / "suite.yaml").write_text(yaml.safe_dump(suite))
    run = Run.open(load_config(fast_config_dir), "grid", results_dir=tmp_path / "results")
    with engine_session(run, "mock", "pre"):  # the preflight checks run against a plain mock
        pass
    engines = fast_config_dir / "engines" / "mock.yaml"
    spec = yaml.safe_load(engines.read_text())
    spec["launch"] += ["--crash-running=2"]
    engines.write_text(yaml.safe_dump(spec))
    folder = run.phase_dir("memory") / "mock"
    folder.mkdir()

    cells = memory.capacity_grid(run, "mock", load_datasets(run), folder)
    statuses = {(c["length"], c["batch"]): c["status"] for c in cells}
    assert statuses == {(512, 1): "ok", (512, 2): "failed", (512, 3): "not run",
                        (1024, 1): "ok", (1024, 2): "failed", (1024, 3): "not run"}
    assert next(c for c in cells if c["batch"] == 3)["reason"] == "row failed"

    lines = (folder / "grid.jsonl").read_text().splitlines()
    assert len(lines) == 6
    memory.capacity_grid(run, "mock", load_datasets(run), folder)
    assert (folder / "grid.jsonl").read_text().splitlines() == lines


def test_lengths_past_the_model_length_are_not_run(run):
    run.cfg.suite["memory"]["grid_lengths"] = [512, 16384]
    run.cfg.suite["memory"]["grid_batches"] = [1]
    folder = run.phase_dir("memory") / "mock"
    folder.mkdir()
    cells = memory.capacity_grid(run, "mock", load_datasets(run), folder)
    assert [(c["length"], c["status"]) for c in cells] == [(512, "ok"), (16384, "not run")]
    assert cells[1]["reason"] == "exceeds max_model_len"


def test_the_suite_writes_every_memory_table_for_a_fake_engine(run):
    probe.execute(run, ["mock"])
    memory.execute(run, ["mock"])
    tables = run.dir / "tables"
    grid = pd.read_csv(tables / "memory_grid.csv")
    assert set(grid.status) == {"ok"} and len(grid) == 4
    assert grid.peak_running.min() >= 1
    exhaustion = pd.read_csv(tables / "memory_exhaustion.csv").iloc[0]
    assert exhaustion.outcome == "completed_all" and bool(exhaustion.recovered)
    assert exhaustion.oversubscription == pytest.approx(8 * 8192 / (4096 * 16))
    kv = pd.read_csv(tables / "kv_util.csv")
    assert len(kv) > 10 and (kv.engine == "mock").all() and kv.kv_util.max() > 0
    assert len(pd.read_csv(tables / "memory_breakdown.csv")) == 6  # the fake's six idle components
    device = pd.read_csv(tables / "memory_device.csv")
    assert {"idle_bytes", "peak_bytes", "peak_source"} <= set(device.columns)
    assert device.peak_source.tolist() == ["kv_util run"]  # no sweep was run for this engine
