"""Background poller for the running server's `/metrics` endpoint, tracking peak/mean
KV-cache utilization over one benchmark run. Scrapes the same Prometheus text any
external monitor would, from the driver process — no direct access to engine
internals needed here.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import httpx
from prometheus_client.parser import text_string_to_metric_families

_METRIC_NAME = "llm_kv_cache_utilization"


@dataclass(slots=True)
class KvStats:
    peak_utilization: float | None = None
    mean_utilization: float | None = None


def _sample(base_url: str) -> float | None:
    try:
        resp = httpx.get(f"{base_url}/metrics", timeout=2.0)
        resp.raise_for_status()
    except httpx.HTTPError:
        return None
    for family in text_string_to_metric_families(resp.text):
        if family.name == _METRIC_NAME:
            for sample in family.samples:
                return sample.value
    return None


class KvUtilizationMonitor:
    """Use as a context manager around one benchmark run; `.stats` is populated on exit."""

    def __init__(self, base_url: str, interval_s: float = 0.2):
        self._base_url = base_url
        self._interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples: list[float] = []
        self.stats = KvStats()

    def __enter__(self) -> "KvUtilizationMonitor":
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=self._interval_s * 5)
        if self._samples:
            self.stats = KvStats(
                peak_utilization=max(self._samples),
                mean_utilization=sum(self._samples) / len(self._samples),
            )

    def _poll_loop(self) -> None:
        while not self._stop.is_set():
            value = _sample(self._base_url)
            if value is not None:
                self._samples.append(value)
            time.sleep(self._interval_s)
