import time

from llm_serving_engine.loadgen.gpu_monitor import GpuMonitor, _sample


def test_gpu_monitor_degrades_gracefully_without_nvidia_smi(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _name: None)
    with GpuMonitor(interval_s=0.01) as monitor:
        pass
    assert monitor.stats.peak_utilization_pct is None
    assert monitor.stats.mean_utilization_pct is None
    assert monitor.stats.peak_memory_used_mb is None
    assert monitor.stats.mean_memory_used_mb is None


def test_gpu_monitor_tracks_peak_and_mean_when_available(monkeypatch):
    readings = [(10.0, 100.0), (50.0, 300.0), (30.0, 200.0)]

    monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        "llm_serving_engine.loadgen.gpu_monitor._sample",
        lambda: readings.pop(0) if readings else None,
    )

    with GpuMonitor(interval_s=0.01) as monitor:
        deadline = time.monotonic() + 2.0
        while readings and time.monotonic() < deadline:
            time.sleep(0.01)  # let the background thread drain the fixed reading sequence

    assert monitor.stats.peak_utilization_pct == 50.0
    assert monitor.stats.mean_utilization_pct == 30.0
    assert monitor.stats.peak_memory_used_mb == 300.0
    assert monitor.stats.mean_memory_used_mb == 200.0


def test_sample_parses_nvidia_smi_csv_output(monkeypatch):
    class _Result:
        stdout = "23, 4096\n"

    monkeypatch.setattr(
        "subprocess.run", lambda *a, **k: _Result()
    )
    assert _sample() == (23.0, 4096.0)


def test_sample_returns_none_when_nvidia_smi_missing(monkeypatch):
    import subprocess

    def _raise(*a, **k):
        raise FileNotFoundError

    monkeypatch.setattr(subprocess, "run", _raise)
    assert _sample() is None
