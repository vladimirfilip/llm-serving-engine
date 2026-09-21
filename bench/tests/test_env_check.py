import shutil
import subprocess
import sys

from bench.run import RESULTS_DIR


def test_env_check_runs_every_preflight_check_for_the_named_engines(
    fast_config_dir, synthetic_datasets, tmp_path
):
    import os

    run_id = "env-check-mock"
    shutil.rmtree(RESULTS_DIR / run_id, ignore_errors=True)
    result = subprocess.run(
        [sys.executable, "-m", "bench", "env-check", "--engines", "mock", "--run-id", run_id,
         "--config-dir", str(fast_config_dir), "--model-path", "bench/tests/fixtures/model"],
        env=os.environ | {"BENCH_DATASETS_DIR": str(synthetic_datasets)},
        capture_output=True, text=True, timeout=300)
    try:
        assert result.returncode == 0, result.stdout[-1500:] + result.stderr[-1500:]
        for check in ("prefix_cache_off", "token_accounting", "no_bursts", "null_server_client"):
            assert f"mock: {check}: pass" in result.stdout
        assert (RESULTS_DIR / run_id / "checks.json").exists()
        assert "package" in result.stdout and "torch" in result.stdout
    finally:
        shutil.rmtree(RESULTS_DIR / run_id, ignore_errors=True)


def test_env_check_without_engines_only_reports_the_machine():
    result = subprocess.run([sys.executable, "-m", "bench", "env-check", "--allow-unlocked"],
                            capture_output=True, text=True, timeout=120,
                            env={"CUDA_VISIBLE_DEVICES": "", **__import__("os").environ})
    assert result.returncode == 0 and "numpy" in result.stdout
