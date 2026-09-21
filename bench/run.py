"""One benchmark run: its id, its results directory, and which phases have finished."""

from __future__ import annotations

import datetime as dt
import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .config import BENCH_DIR, REPO_ROOT, Config

RESULTS_DIR = BENCH_DIR / "results"
# A dirty tree here does not change what the engine under test is.
NOT_ENGINE_SOURCE = ("bench/results/", "bench/datasets/", "bench/published/", "README.md")


@dataclass(frozen=True, slots=True)
class GitInfo:
    commit: str
    short: str
    dirty: bool


def git_info(repo: Path = REPO_ROOT) -> GitInfo:
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True).stdout

    lines = [line for line in git("status", "--porcelain").splitlines() if line.strip()]
    dirty = any(not line[3:].startswith(NOT_ENGINE_SOURCE) for line in lines)
    commit = git("rev-parse", "HEAD").strip()
    return GitInfo(commit, commit[:7], dirty)


def new_run_id(short_hash: str, now: dt.datetime | None = None) -> str:
    return f"{(now or dt.datetime.now()).strftime('%Y%m%d-%H%M%S')}-{short_hash}"


@dataclass(slots=True)
class Run:
    id: str
    dir: Path
    cfg: Config
    force: bool = False
    allow_unlocked: bool = False
    client_procs: int = 1
    _status: dict = field(default_factory=dict)

    @classmethod
    def open(cls, cfg: Config, run_id: str | None = None, results_dir: Path = RESULTS_DIR,
             **flags) -> Run:
        run_id = run_id or new_run_id(git_info().short)
        run = cls(run_id, results_dir / run_id, cfg, **flags)
        for sub in ("logs", "gpu_monitor", "tables", "plots", "config_snapshot"):
            (run.dir / sub).mkdir(parents=True, exist_ok=True)
        run._status = json.loads(run.status_path.read_text()) if run.status_path.exists() else {}
        run.snapshot_config()
        return run

    @property
    def status_path(self) -> Path:
        return self.dir / "status.json"

    def snapshot_config(self) -> None:
        snap = self.dir / "config_snapshot"
        root = self.cfg.dir
        for path in [*root.glob("*.yaml"), *(root / "engines").glob("*.yaml")]:
            target = snap / path.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)

    def phase_dir(self, phase: str) -> Path:
        path = self.dir / phase
        path.mkdir(parents=True, exist_ok=True)
        return path

    def is_done(self, phase: str) -> bool:
        return not self.force and (self.dir / phase / "DONE").exists()

    def mark_done(self, phase: str) -> None:
        (self.phase_dir(phase) / "DONE").write_text(dt.datetime.now().isoformat())
        self.set_status(phase, "done")

    def set_status(self, phase: str, state: str, note: str = "") -> None:
        self._status[phase] = {"state": state, "note": note}
        self.status_path.write_text(json.dumps(self._status, indent=2))

    def aborted(self, phase: str) -> list[str]:
        """Engines whose part of `phase` a failed MUST check aborted."""
        prefix = f"{phase}:"
        return [key.removeprefix(prefix) for key, v in self._status.items()
                if key.startswith(prefix) and v["state"] == "aborted"]

    def log_path(self, engine: str, phase: str) -> Path:
        return self.dir / "logs" / f"{engine}_{phase}.log"

    def monitor_path(self, engine: str, phase: str) -> Path:
        return self.dir / "gpu_monitor" / f"{engine}_{phase}.parquet"
