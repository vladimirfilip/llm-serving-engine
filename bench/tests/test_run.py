import datetime as dt
import json

from bench.config import load_config
from bench.run import Run, new_run_id


def test_run_id_carries_the_timestamp_and_engine_commit():
    now = dt.datetime(2026, 9, 21, 13, 5, 9)
    assert new_run_id("3825144", now) == "20260921-130509-3825144"


def test_run_layout_and_config_snapshot(tmp_path):
    run = Run.open(load_config(), "r1", results_dir=tmp_path)
    for sub in ("logs", "gpu_monitor", "tables", "plots", "config_snapshot"):
        assert (run.dir / sub).is_dir()
    assert (run.dir / "config_snapshot" / "suite.yaml").exists()


def test_a_finished_phase_is_skipped_until_forced(tmp_path):
    cfg = load_config()
    run = Run.open(cfg, "r1", results_dir=tmp_path)
    assert not run.is_done("sweep")
    run.mark_done("sweep")
    assert Run.open(cfg, "r1", results_dir=tmp_path).is_done("sweep")
    assert not Run.open(cfg, "r1", results_dir=tmp_path, force=True).is_done("sweep")


def test_status_survives_reopening(tmp_path):
    cfg = load_config()
    run = Run.open(cfg, "r1", results_dir=tmp_path)
    run.set_status("kernels", "failed", "no GPU")
    reopened = Run.open(cfg, "r1", results_dir=tmp_path)
    assert json.loads(reopened.status_path.read_text())["kernels"] == {
        "state": "failed", "note": "no GPU"}
