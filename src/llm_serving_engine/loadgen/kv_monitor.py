"""Background poller for the running server's `/metrics` endpoint over one benchmark run:
KV-cache utilization, and how many sequences were preempted for lack of KV blocks. Scrapes
the same Prometheus text any external monitor would, from the driver process.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import httpx
from prometheus_client.parser import text_string_to_metric_families

_UTILIZATION = "llm_kv_cache_utilization"
_PREEMPTIONS = "llm_preemptions_total"


@dataclass(slots=True)
class KvStats:
    peak_utilization: float | None = None
    mean_utilization: float | None = None
    preemptions: int | None = None


def _sample(base_url: str) -> tuple[float, float] | None:
    """One (utilization, preemptions counter) reading, or None if the scrape fails."""
    try:
        resp = httpx.get(f"{base_url}/metrics", timeout=2.0)
        resp.raise_for_status()
    except httpx.HTTPError:
        return None
    values = {
        sample.name: sample.value
        for family in text_string_to_metric_families(resp.text)
        for sample in family.samples
    }
    if _UTILIZATION not in values or _PREEMPTIONS not in values:
        return None
    return values[_UTILIZATION], values[_PREEMPTIONS]


class KvCacheMonitor:
    """Use as a context manager around one benchmark run; `.stats` is populated on exit.
    Utilization is polled, so a peak shorter than `interval_s` can go unseen; preemptions
    are a counter, so their count over the run is exact."""

    def __init__(self, base_url: str, interval_s: float = 0.2):
        self._base_url = base_url
        self._interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._utilizations: list[float] = []
        self._preemption_counts: list[float] = []
        self.stats = KvStats()

    def __enter__(self) -> KvCacheMonitor:
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=self._interval_s * 5)
        final = _sample(self._base_url)
        if final is not None:
            self._utilizations.append(final[0])
            self._preemption_counts.append(final[1])
        if self._utilizations:
            self.stats = KvStats(
                peak_utilization=max(self._utilizations),
                mean_utilization=sum(self._utilizations) / len(self._utilizations),
                preemptions=int(self._preemption_counts[-1] - self._preemption_counts[0]),
            )

    def _poll_loop(self) -> None:
        while not self._stop.is_set():
            reading = _sample(self._base_url)
            if reading is not None:
                self._utilizations.append(reading[0])
                self._preemption_counts.append(reading[1])
            time.sleep(self._interval_s)
