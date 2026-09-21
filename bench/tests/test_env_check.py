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


def test_env_check_without_engines_reports_the_machine_and_never_touches_the_card(
    fast_config_dir,
):
    import yaml

    hardware = yaml.safe_load((fast_config_dir / "hardware.yaml").read_text())
    hardware["gpu_clock_mhz"] = None  # nothing to lock, so lock_clocks leaves the card alone
    (fast_config_dir / "hardware.yaml").write_text(yaml.safe_dump(hardware))
    result = subprocess.run(
        [sys.executable, "-m", "bench", "env-check", "--allow-unlocked", "--config-dir",
         str(fast_config_dir)], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout[-800:] + result.stderr[-800:]
    assert "numpy" in result.stdout
    assert "clocks locked: False gpu_clock_mhz is not set" in result.stdout or (
        "no NVIDIA driver" in result.stdout)


def test_a_clock_lock_that_cannot_be_verified_is_reported_unlocked_not_raised(monkeypatch):
    from bench import env

    monkeypatch.setattr(env, "nvidia_smi", lambda *a: type("R", (), {"returncode": 0})())
    monkeypatch.setattr(env, "run_task", lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("No CUDA GPUs are available")))
    lock = env.lock_clocks({"gpu_index": 0, "gpu_clock_mhz": 1900, "mem_clock_mhz": None,
                            "power_limit_w": None})
    assert not lock.locked and "No CUDA GPUs are available" in lock.error
