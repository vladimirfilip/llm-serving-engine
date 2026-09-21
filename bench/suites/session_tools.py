"""Helpers for suites that watch an engine while loading it, or expect to crash it."""

from __future__ import annotations

import contextlib
import threading
import time

from ..engines.generic import GenericAdapter
from ..run import Run
from .common import Session, engine_session


class StatsPoller:
    """Polls `adapter.stats()` on a thread at `hz`, timestamped with `perf_counter`. Samples
    are dicts, or absent for an engine with no stats."""

    def __init__(self, adapter: GenericAdapter, hz: float):
        self.adapter, self.period = adapter, 1 / hz
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def __enter__(self) -> StatsPoller:
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()

    def _loop(self) -> None:
        next_tick = time.perf_counter()
        while not self._stop.is_set():
            stats = self.adapter.stats()
            if stats is not None:
                self.samples.append({"t": time.perf_counter(), **stats})
            next_tick += self.period
            self._stop.wait(max(0.0, next_tick - time.perf_counter()))

    def between(self, t0: float, t1: float) -> list[dict]:
        return [s for s in self.samples if t0 <= s["t"] <= t1]


class Relauncher:
    """An engine session that starts a fresh engine when the running one has exited, for
    phases that push an engine until it fails."""

    def __init__(self, run: Run, engine: str, phase: str):
        self.run, self.engine, self.phase = run, engine, phase
        self._stack = contextlib.ExitStack()
        self._session: Session | None = None
        self.launches = 0

    def __enter__(self) -> Relauncher:
        return self

    def __exit__(self, *exc) -> None:
        self._stack.close()

    def session(self) -> Session:
        if self._session is None or self._session.adapter.proc.exited():
            self._stack.close()
            self._stack = contextlib.ExitStack()
            self._session = self._stack.enter_context(
                engine_session(self.run, self.engine, self.phase))
            self.launches += 1
        return self._session
