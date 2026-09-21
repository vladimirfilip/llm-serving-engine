import pandas as pd
import pytest

from bench.env import HW_THERMAL, SW_POWER_CAP, GpuMonitor


def monitor_with(rows: list[dict], locked_clock_mhz: float | None = 1900) -> GpuMonitor:
    monitor = GpuMonitor(0, locked_clock_mhz=locked_clock_mhz)
    monitor._rows = rows
    return monitor


def status(t: float, reasons: int, sm: float = 1900) -> dict:
    return {"t": t, "event_reasons": reasons, "sm_clock_mhz": sm, "energy_mj": 0.0}


def test_energy_is_interpolated_between_counter_samples():
    rows = [{"t": t, "energy_mj": 1000.0 * t} for t in (0.0, 0.1, 0.2, 0.3)]
    assert monitor_with(rows).energy_j(0.05, 0.25) == pytest.approx(0.2)


def test_energy_is_none_when_the_window_leaves_the_samples():
    rows = [{"t": t, "energy_mj": 1000.0 * t} for t in (1.0, 2.0)]
    assert monitor_with(rows).energy_j(0.0, 1.5) is None


def test_a_locked_gpu_reporting_the_power_cap_bit_is_not_throttled():
    rows = [status(t, SW_POWER_CAP) for t in range(100)]
    assert not monitor_with(rows).throttled(0, 99)


def test_a_power_cap_that_costs_clock_is_throttled():
    rows = [status(t, SW_POWER_CAP, sm=1500) for t in range(100)]
    assert monitor_with(rows).throttled(0, 99)


def test_thermal_slowdown_over_one_percent_of_samples_is_throttled():
    ok = [status(t, 0) for t in range(98)]
    assert not monitor_with([*ok, status(98, HW_THERMAL), status(99, 0)]).throttled(0, 99)
    assert monitor_with([*ok, status(98, HW_THERMAL), status(99, HW_THERMAL)]).throttled(0, 99)


def test_unavailable_monitor_reports_unknown_without_a_driver():
    monitor = GpuMonitor(0)
    monitor._nvml = None
    with monitor:
        pass
    assert isinstance(monitor.frame(), pd.DataFrame) and monitor.frame().empty
    assert monitor.peak_memory_bytes(0, 1) is None
    assert monitor.throttled(0, 1) is None
    assert monitor.energy_j(0, 1) is None


def test_a_window_without_a_status_sample_is_unknown_not_clear():
    rows = [status(t, 0) for t in range(10)]
    assert monitor_with(rows).throttled(20, 30) is None
    only_energy = [{"t": t, "energy_mj": 0.0} for t in range(10)]
    assert monitor_with(only_energy).throttled(0, 9) is None
