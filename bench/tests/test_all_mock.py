"""The end-to-end smoke test: every phase, then the report, against the fake engine with no
GPU work and no model files beyond a config.json."""

import json
import os
import shutil
import subprocess
import sys

from bench.run import RESULTS_DIR

ENV_KEYS = {"CUDA_VISIBLE_DEVICES": ""}


def test_bench_all_completes_on_the_mock_and_reports_what_it_skipped(
    fast_config_dir, tmp_path, monkeypatch
):
    run_id = "smoke-all-mock"
    shutil.rmtree(RESULTS_DIR / run_id, ignore_errors=True)
    env = os.environ | {"BENCH_DATASETS_DIR": str(tmp_path / "datasets")} | ENV_KEYS
    result = subprocess.run(
        [sys.executable, "-m", "bench", "all", "--engines", "mock", "--quick", "--run-id", run_id,
         "--config-dir", str(fast_config_dir), "--model-path", "bench/tests/fixtures/model"],
        env=env, capture_output=True, text=True, timeout=1500)
    run_dir = RESULTS_DIR / run_id
    try:
        assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-3000:]
        status = json.loads((run_dir / "status.json").read_text())
        assert status["sweep"]["state"] == "done" and status["memory"]["state"] == "done"
        assert status["kernels"]["state"] == "skipped" and "ours" in status["kernels"]["note"]
        assert status["ablation"]["state"] == "skipped"
        assert status["precision"]["state"] == "skipped"
        report = (run_dir / "report.md").read_text()
        assert "Synthetic datasets" in report and "Quick run" in report
        assert "kernels: skipped" in report and "ablation: skipped" in report
        assert "| mock |" in report
        assert (run_dir / "plots" / "p01_throughput_latency_sharegpt.png").exists()
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
