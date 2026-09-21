"""A server subprocess: launch in its own session, log to a file, stop by process group."""

from __future__ import annotations

import ctypes
import os
import signal
import socket
import subprocess
import time
from pathlib import Path

from .. import env as bench_env

PR_SET_PDEATHSIG = 1
TERM_GRACE_S = 30
GPU_FREE_TIMEOUT_S = 60


def die_with_parent() -> None:
    """Runs in the child before exec: SIGTERM it if the harness dies, so a killed run does not
    leave an engine holding the GPU."""
    ctypes.CDLL("libc.so.6").prctl(PR_SET_PDEATHSIG, signal.SIGTERM)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ServerProcess:
    def __init__(self, argv: list[str], env: dict[str, str], log_path: Path,
                 cpus: str | None = None, wait_gpu_free: bool = True):
        self.argv = ["taskset", "-c", cpus, *argv] if cpus else argv
        self.env = env
        self.log_path = log_path
        self.wait_gpu_free = wait_gpu_free
        self.proc: subprocess.Popen | None = None
        self.started_at = 0.0

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log = self.log_path.open("ab")
        self.started_at = time.perf_counter()
        self.proc = subprocess.Popen(self.argv, env={**os.environ, **self.env}, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True,
                                     preexec_fn=die_with_parent)

    @property
    def pid(self) -> int:
        return self.proc.pid

    def exited(self) -> bool:
        return self.proc is not None and self.proc.poll() is not None

    def log_tail(self, lines: int = 100) -> str:
        return "\n".join(self.log_path.read_text(errors="replace").splitlines()[-lines:])

    def stop(self) -> None:
        """SIGTERM the group, SIGKILL after the grace period, then wait for the GPU to be free."""
        if self.proc is None:
            return
        for sig, wait in ((signal.SIGTERM, TERM_GRACE_S), (signal.SIGKILL, 10)):
            if self.proc.poll() is not None:
                break
            try:
                os.killpg(self.proc.pid, sig)
            except ProcessLookupError:
                break
            try:
                self.proc.wait(timeout=wait)
            except subprocess.TimeoutExpired:
                continue
        self.proc = None
        if self.wait_gpu_free and bench_env.gpu_available() and not bench_env.wait_gpu_free(
            GPU_FREE_TIMEOUT_S
        ):
            raise RuntimeError("GPU still has compute processes 60 s after shutdown")
