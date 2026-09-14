"""Background GPU sampler for a benchmark run: peak/mean utilization% and memory-used,
via `nvidia-smi` (whole-device, so it sees the server subprocess's usage even though
this runs in the benchmark driver's own process). Without `nvidia-smi` every stat is
`None`, so a run still completes off-GPU.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from dataclasses import dataclass


@dataclass(slots=True)
class GpuStats:
    peak_utilization_pct: float | None = None
    mean_utilization_pct: float | None = None
    peak_memory_used_mb: float | None = None
    mean_memory_used_mb: float | None = None


def _sample() -> tuple[float, float] | None:
    """One (utilization_pct, memory_used_mb) reading, or None if nvidia-smi fails."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5.0, check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    first_line = out.stdout.strip().splitlines()[0]
    util_str, mem_str = first_line.split(",")
    return float(util_str), float(mem_str)


class GpuMonitor:
    """Use as a context manager around one benchmark run; `.stats` is populated on exit."""

    def __init__(self, interval_s: float = 0.2):
        self._interval_s = interval_s
        self._available = shutil.which("nvidia-smi") is not None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._utilizations: list[float] = []
        self._memories_mb: list[float] = []
        self.stats = GpuStats()

    def __enter__(self) -> "GpuMonitor":
        if self._available:
            self._stop.clear()
            self._thread = threading.Thread(target=self._poll_loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=self._interval_s * 5)
        if self._utilizations:
            self.stats = GpuStats(
                peak_utilization_pct=max(self._utilizations),
                mean_utilization_pct=sum(self._utilizations) / len(self._utilizations),
                peak_memory_used_mb=max(self._memories_mb),
                mean_memory_used_mb=sum(self._memories_mb) / len(self._memories_mb),
            )

    def _poll_loop(self) -> None:
        while not self._stop.is_set():
            reading = _sample()
            if reading is not None:
                util, mem_mb = reading
                self._utilizations.append(util)
                self._memories_mb.append(mem_mb)
            time.sleep(self._interval_s)
