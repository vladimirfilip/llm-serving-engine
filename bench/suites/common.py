"""What every suite shares: the per-run environment record, tuned launches, and an engine
session that launches, checks, monitors and shuts an engine down."""

from __future__ import annotations

import contextlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .. import env as bench_env
from ..client.stream import Target
from ..config import EngineSpec, load_engine
from ..engines.base import CheckFailed, Launch
from ..engines.generic import GenericAdapter, make_adapter
from ..run import Run
from ..workloads.workloads import DATASETS_DIR, Datasets


class SuiteSkipped(Exception):
    """A suite that has nothing to run here, with the reason the report will show."""


@dataclass(slots=True)
class Session:
    run: Run
    spec: EngineSpec
    adapter: GenericAdapter
    monitor: bench_env.GpuMonitor
    launch: Launch
    phase: str
    ready_s: float
    client_procs: int

    def target(self, logprobs: bool = False, timeout_s: float | None = None) -> Target:
        sweep = self.run.cfg.suite["load_sweep"]
        return Target(self.adapter.base_url(), self.adapter.request_model, self.adapter.extra_body,
                      self.adapter.accepts_token_ids, timeout_s or sweep["timeout_s"],
                      self.run.cfg.model_path, logprobs)

    @property
    def client_cpus(self) -> str | None:
        return self.run.cfg.hardware["cpu_affinity"]["client"]

    @property
    def server_cpus(self) -> str | None:
        return self.run.cfg.hardware["cpu_affinity"]["server"]

    @property
    def cooldown_s(self) -> float:
        """Between load points, so thermal drift is the same across engines; a fake one has none."""
        return 10.0 if self.spec.uses_gpu else 0.0


def env_path(run: Run) -> Path:
    return run.dir / "env.json"


def load_env(run: Run) -> dict:
    return json.loads(env_path(run).read_text())


def ensure_env(run: Run, uses_gpu: bool = True) -> dict:
    """Writes `env.json`, and for an engine that uses the GPU locks the clocks and measures
    bandwidth first. The GPU step is redone if the file was written by engines that had none."""
    if env_path(run).exists():
        env = load_env(run)
        if not uses_gpu or env["gpu_prepared"]:
            return env
    hw = run.cfg.hardware
    prepare_gpu = uses_gpu and bench_env.gpu_available()
    if prepare_gpu:
        lock = bench_env.lock_clocks(hw)
        if not lock.locked and not run.allow_unlocked:
            raise CheckFailed(f"clocks not locked: {lock.error} (pass --allow-unlocked to proceed)")
        probe = bench_env.measure_bandwidth()
    else:
        reason = "no NVIDIA driver" if uses_gpu else "no engine in this run uses the GPU"
        lock, probe = bench_env.ClockLock(False, reason), None
    env = bench_env.capture_env(run.cfg, lock, probe) | {"gpu_prepared": prepare_gpu}
    env_path(run).write_text(json.dumps(env, indent=2))
    return env


def tuned_path(run: Run) -> Path:
    return run.dir / "tune" / "tuned.json"


def tuned_launch(run: Run, engine: str) -> Launch:
    """The token budget `bench tune` picked, or the engine's default when it was not tuned."""
    if tuned_path(run).exists():
        entry = json.loads(tuned_path(run).read_text()).get(engine)
        if entry:
            return Launch(token_budget=entry["token_budget"])
    return Launch(token_budget=load_engine(engine, run.cfg.dir).default_token_budget)


def load_datasets(run: Run) -> Datasets:
    meta = json.loads((datasets_dir() / "meta.json").read_text())
    return Datasets.load(meta["bos_token_id"], datasets_dir())


def datasets_dir() -> Path:
    """`bench/datasets`, or `$BENCH_DATASETS_DIR` for tests and alternate builds."""
    return Path(os.environ.get("BENCH_DATASETS_DIR", DATASETS_DIR))


@contextlib.contextmanager
def guarded(run: Run, phase: str, engine: str) -> Iterator[None]:
    """A failed MUST check aborts this engine's phase, is recorded, and lets the rest go on."""
    try:
        yield
    except CheckFailed as e:
        run.set_status(f"{phase}:{engine}", "aborted", str(e))


@contextlib.contextmanager
def engine_session(run: Run, engine: str, phase: str, launch: Launch | None = None,
                   checks: bool = True) -> Iterator[Session]:
    """Launch `engine`, wait until it serves, run the preflight checks once per run, and shut it
    down on exit. The GPU monitor covers the whole session."""
    from .checks import ensure_checks, require_idle_gpu

    spec = load_engine(engine, run.cfg.dir)
    env = ensure_env(run, spec.uses_gpu)
    adapter = make_adapter(spec, run.cfg, run)
    launch = launch or tuned_launch(run, engine)
    locked = env.get("clocks_locked") and env["hardware"]["gpu_clock_mhz"]
    monitor = bench_env.GpuMonitor(run.cfg.hardware["gpu_index"], run.monitor_path(engine, phase),
                                   locked_clock_mhz=locked or None)
    if spec.uses_gpu:
        require_idle_gpu(run, engine)
    adapter.launch(launch, phase)
    try:
        ready_s = adapter.wait_ready()
        monitor.server_pid = adapter.proc.pid
        with monitor:
            session = Session(run, spec, adapter, monitor, launch, phase, ready_s,
                              run.client_procs)
            if checks:
                ensure_checks(session)
            yield session
    finally:
        adapter.shutdown()
