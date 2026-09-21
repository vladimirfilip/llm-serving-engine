"""The machine under test: package inventory, clock lock, GPU monitor and bandwidth probe.
Everything here degrades to "unavailable" on a machine with no NVIDIA driver, so the mock
engine's runs work on a CPU-only box."""

from __future__ import annotations

import dataclasses
import importlib.metadata as metadata
import itertools
import os
import platform
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

from .config import Config
from .gpu_tasks import run_task
from .modelspec import ModelSpec
from .run import git_info

# (import name on PyPI, needed for, what happens when it is missing)
PACKAGES: list[tuple[str, str, str]] = [
    ("numpy", "core harness", "the harness does not start"),
    ("pandas", "core harness", "the harness does not start"),
    ("pyarrow", "core harness", "the harness does not start"),
    ("pyyaml", "core harness", "the harness does not start"),
    ("aiohttp", "core harness", "the harness does not start"),
    ("orjson", "core harness", "the harness does not start"),
    ("matplotlib", "plots", "the harness does not start"),
    ("scipy", "scheduler metrics", "the harness does not start"),
    ("psutil", "gpu monitor", "the harness does not start"),
    ("nvidia-ml-py", "gpu monitor", "the harness does not start"),
    ("torch", "reference model, kernels", "the harness does not start"),
    ("transformers", "reference model, tokenizer", "the harness does not start"),
    ("huggingface_hub", "model and dataset download", "the harness does not start"),
    ("datasets", "prepare-data (WikiText-2)", "prepare-data fails"),
    ("uvloop", "client event loop", "falls back to asyncio, noted in the report"),
    ("lm-eval", "GSM8K", "GSM8K skipped, gate 4 reported as not evaluated"),
    ("flash-attn", "kernel contender", "that contender is skipped and listed in the report"),
    ("flashinfer-python", "kernel contender", "that contender is skipped and listed in the report"),
]

# nvmlClocksEventReason bits
SW_POWER_CAP, HW_SLOWDOWN, SW_THERMAL, HW_THERMAL = 0x4, 0x8, 0x20, 0x40
THROTTLE_MASK = HW_SLOWDOWN | SW_THERMAL | HW_THERMAL
CLOCK_TOLERANCE = 0.02
MONITOR_HZ = 10
STATUS_EVERY = 10  # ticks between the 1 Hz status columns


def package_versions() -> dict[str, str | None]:
    def version(name: str) -> str | None:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return None

    return {name: version(name) for name, _, _ in PACKAGES}


def format_package_table(versions: dict[str, str | None]) -> str:
    rows = [f"{'package':<20}{'status':<10}{'version':<16}needed for / if missing"]
    for name, needed, missing in PACKAGES:
        v = versions[name]
        state = "present" if v else "MISSING"
        rows.append(f"{name:<20}{state:<10}{v or '-':<16}{needed} / {missing}")
    return "\n".join(rows)


def _nvml():
    """The initialised pynvml module, or None when there is no driver."""
    try:
        import pynvml

        pynvml.nvmlInit()
        return pynvml
    except Exception:
        return None


def gpu_available() -> bool:
    return _nvml() is not None


def nvidia_smi(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["nvidia-smi", *args], capture_output=True, text=True)


def compute_pids() -> list[int]:
    out = nvidia_smi("--query-compute-apps=pid", "--format=csv,noheader").stdout
    return [int(line) for line in out.split() if line.strip().isdigit()]


def wait_gpu_free(timeout_s: float) -> bool:
    deadline = time.perf_counter() + timeout_s
    while compute_pids():
        if time.perf_counter() > deadline:
            return False
        time.sleep(0.5)
    return True


def gpu_utilization_pct(gpu_index: int, samples: int = 5) -> float:
    nvml = _nvml()
    handle = nvml.nvmlDeviceGetHandleByIndex(gpu_index)
    readings = []
    for _ in range(samples):
        readings.append(nvml.nvmlDeviceGetUtilizationRates(handle).gpu)
        time.sleep(0.1)
    return max(readings)


@dataclass(slots=True)
class ClockLock:
    locked: bool
    error: str = ""
    observed_mhz: int | None = None  # the SM clock the lock settled on under load


def lock_clocks(hw: dict) -> ClockLock:
    """Applies the clock and power settings in `hardware.yaml`, then reads the SM clock back
    under load. A GPU whose settings are unset or refused is reported as unlocked."""
    gpu = str(hw["gpu_index"])
    if hw["gpu_clock_mhz"] is None:
        return ClockLock(False, "gpu_clock_mhz is not set")
    commands = [["-pm", "1"], ["-lgc", f"{hw['gpu_clock_mhz']},{hw['gpu_clock_mhz']}"]]
    if hw.get("mem_clock_mhz"):
        commands.append(["-lmc", f"{hw['mem_clock_mhz']},{hw['mem_clock_mhz']}"])
    if hw.get("power_limit_w"):
        commands.append(["-pl", str(hw["power_limit_w"])])
    for command in commands:
        result = nvidia_smi("-i", gpu, *command)
        if result.returncode != 0:
            return ClockLock(False, (result.stdout + result.stderr).strip())
    try:
        return ClockLock(**run_task("verify-clock", hw=hw))
    except RuntimeError as e:
        return ClockLock(False, f"could not verify the clock lock: {str(e)[-300:]}")


def verify_clock_lock(hw: dict) -> ClockLock:
    """The driver settles a lock on the nearest clock bin, so the check is a steady SM clock
    close to the target, not an exact match."""
    import torch

    nvml = _nvml()
    handle = nvml.nvmlDeviceGetHandleByIndex(hw["gpu_index"])
    x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    clocks = []
    for _ in range(20):
        for _ in range(20):
            x @ x
        torch.cuda.synchronize()
        clocks.append(nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_SM))
    target, observed = hw["gpu_clock_mhz"], int(np.median(clocks))
    off_target = abs(observed - target) > CLOCK_TOLERANCE * target
    if off_target or max(clocks) - min(clocks) > 0.01 * target:
        return ClockLock(False, f"SM clock reads {min(clocks)}-{max(clocks)} MHz under load, "
                                f"locked to {target}", observed)
    return ClockLock(True, "", observed)


def reset_clocks(hw: dict) -> None:
    gpu = str(hw["gpu_index"])
    nvidia_smi("-i", gpu, "-rgc")
    nvidia_smi("-i", gpu, "-rmc")


class GpuMonitor:
    """Samples NVML on a thread. Timestamps are `perf_counter`, the clock the client stamps
    requests with, so a window `[t0, t1]` from a run indexes straight into the samples."""

    def __init__(self, gpu_index: int, path: Path | None = None,
                 locked_clock_mhz: float | None = None):
        self.gpu_index = gpu_index
        self.path = path
        self.locked_clock_mhz = locked_clock_mhz
        self.server_pid: int | None = None
        self._nvml = _nvml()
        self._rows: list[dict] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def available(self) -> bool:
        return self._nvml is not None

    def __enter__(self) -> GpuMonitor:
        if self.available:
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, name="gpu-monitor", daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        if self._thread is not None:
            self._stop.set()
            self._thread.join()
            if self.path is not None:
                self.frame().to_parquet(self.path)

    def _loop(self) -> None:
        nvml = self._nvml
        handle = nvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
        try:
            reasons = nvml.nvmlDeviceGetCurrentClocksEventReasons
        except AttributeError:  # bindings older than the rename
            reasons = nvml.nvmlDeviceGetCurrentClocksThrottleReasons
        period = 1 / MONITOR_HZ
        next_tick = time.perf_counter()
        for tick in itertools.count():
            if self._stop.is_set():
                return
            row = {"t": time.perf_counter(),
                   "energy_mj": nvml.nvmlDeviceGetTotalEnergyConsumption(handle)}
            if tick % STATUS_EVERY == 0:
                row |= {
                    "sm_clock_mhz": nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_SM),
                    "mem_clock_mhz": nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_MEM),
                    "power_w": nvml.nvmlDeviceGetPowerUsage(handle) / 1000,
                    "temp_c": nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU),
                    "event_reasons": int(reasons(handle)),
                    "mem_used_bytes": nvml.nvmlDeviceGetMemoryInfo(handle).used,
                    "server_rss_bytes": self._server_rss(),
                }
            with self._lock:
                self._rows.append(row)
            next_tick += period
            time.sleep(max(0.0, next_tick - time.perf_counter()))

    def _server_rss(self) -> int | None:
        if self.server_pid is None:
            return None
        try:
            root = psutil.Process(self.server_pid)
            return sum(p.memory_info().rss for p in [root, *root.children(recursive=True)])
        except psutil.NoSuchProcess:
            return None

    def frame(self) -> pd.DataFrame:
        with self._lock:
            return pd.DataFrame(self._rows)

    def _within(self, t0: float, t1: float) -> pd.DataFrame:
        df = self.frame()
        return df[(df.t >= t0) & (df.t <= t1)] if len(df) else df

    def energy_j(self, t0: float, t1: float) -> float | None:
        """Energy over [t0, t1], the counter interpolated between its 10 Hz samples."""
        df = self.frame()
        if len(df) < 2 or df.t.iloc[0] > t0 or df.t.iloc[-1] < t1:
            return None
        e0, e1 = np.interp([t0, t1], df.t.to_numpy(), df.energy_mj.to_numpy())
        return float(e1 - e0) / 1000

    def throttled(self, t0: float, t1: float) -> bool | None:
        """More than 1% of the window's status samples show a thermal or hardware slowdown,
        or a software power cap with the SM clock below the locked clock. A locked GPU
        reports the power-cap bit continuously without losing clock, so the bit alone is not
        a throttle. None when the window holds no status sample: unknown, not clear."""
        window = self._within(t0, t1)
        if "event_reasons" not in window:
            return None
        status = window.dropna(subset=["event_reasons"])
        if status.empty:
            return None
        reasons = status.event_reasons.astype(int)
        bad = (reasons & THROTTLE_MASK) > 0
        if self.locked_clock_mhz:
            bad |= ((reasons & SW_POWER_CAP) > 0) & (
                status.sm_clock_mhz < (1 - CLOCK_TOLERANCE) * self.locked_clock_mhz
            )
        return float(bad.mean()) > 0.01

    def peak_memory_bytes(self, t0: float, t1: float) -> int | None:
        used = self._within(t0, t1).mem_used_bytes.dropna() if self.available else []
        return int(max(used)) if len(used) else None


def measure_bandwidth() -> dict[str, float]:
    """`bandwidth_probe` in a child process, so this one never holds a CUDA context."""
    return run_task("bwprobe")


def bandwidth_probe(alloc_gib: int = 4, warmup: int = 10, timed: int = 50) -> dict[str, float]:
    """Best-of read bandwidth from a bf16 reduction and copy bandwidth from a device-to-device
    copy (bytes read plus bytes written), in GB/s. Run with clocks locked and no engine up."""
    import torch

    n = alloc_gib * 2**30 // 2
    x = torch.ones(n, device="cuda", dtype=torch.bfloat16)
    y = torch.empty_like(x)

    def best_seconds(op) -> float:
        for _ in range(warmup):
            op()
        times = []
        for _ in range(timed):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            op()
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end) / 1000)
        return min(times)

    return {
        "bw_read_gbs": x.numel() * 2 / best_seconds(x.sum) / 1e9,
        "bw_copy_gbs": 2 * x.numel() * 2 / best_seconds(lambda: y.copy_(x)) / 1e9,
    }


def gpu_facts(gpu_index: int) -> dict:
    nvml = _nvml()
    if nvml is None:
        return {"available": False}
    handle = nvml.nvmlDeviceGetHandleByIndex(gpu_index)
    name = nvml.nvmlDeviceGetName(handle)
    return {
        "available": True,
        "name": name.decode() if isinstance(name, bytes) else name,
        "driver": nvml.nvmlSystemGetDriverVersion(),
        "memory_total_bytes": nvml.nvmlDeviceGetMemoryInfo(handle).total,
    }


def baseline_versions(cfg: Config) -> dict[str, str | None]:
    """The installed version of each engine that runs from its own environment, read by asking
    that environment's interpreter."""
    import yaml

    from .config import expand, load_engine

    versions: dict[str, str | None] = {}
    for path in sorted((cfg.dir / "engines").glob("*.yaml")):
        if "launch" not in yaml.safe_load(path.read_text()):
            continue  # the ablation steps file is not an engine
        spec = load_engine(path.stem, cfg.dir)
        if not (spec.python and spec.package):
            continue
        python = expand(spec.python, {})
        code = f"import importlib.metadata as m; print(m.version({spec.package!r}))"
        result = (subprocess.run([python, "-c", code], capture_output=True, text=True)
                  if Path(python).exists() else None)
        versions[spec.name] = result.stdout.strip() if result and result.returncode == 0 else None
    return versions


def capture_env(cfg: Config, clock_lock: ClockLock, probe: dict | None) -> dict:
    """The resolved machine and fairness settings written to `env.json`."""
    import torch

    hw, model = cfg.hardware, cfg.model
    git = git_info()
    spec = ModelSpec.from_dir(cfg.model_path, model["dtype"])
    env = {
        "git_commit": git.commit,
        "git_dirty": git.dirty,
        "clocks_locked": clock_lock.locked,
        "clock_lock_error": clock_lock.error,
        "hostname": socket.gethostname(),
        "os": platform.platform(),
        "cpu_count": os.cpu_count(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": gpu_facts(hw["gpu_index"]),
        "hardware": hw,
        "model": model | {"path": cfg.model_path, "spec": dataclasses.asdict(spec)},
        "packages": package_versions(),
        "baselines": baseline_versions(cfg),
        "fairness": {
            "F1": {"checkpoint": cfg.model_path, "dtype": model["dtype"],
                   "gpu_clock_mhz": hw["gpu_clock_mhz"], "mem_clock_mhz": hw["mem_clock_mhz"]},
            "F2": "prefix caching off, verified by check 5.1(c)",
            "F3": "speculative decoding off, verified by check 5.1(e)",
            "F4": {k: model[k] for k in ("max_model_len", "max_num_seqs", "gpu_mem_util")}
                  | {"cuda_graphs": "on wherever the engine has them"},
            "F5": "chunked prefill on wherever supported; token budget tuned per engine",
            "F6": "every engine tuned by `bench tune`; launch commands saved in the results",
            "F7": "one engine process at a time; nvidia-smi shows no compute process at launch",
        },
        "quick": cfg.quick,
    }
    if clock_lock.observed_mhz and hw.get("boost_clock_mhz"):
        env["peak_tflops_bf16_at_locked_clock"] = (
            hw["peak_tflops_bf16_dense"] * clock_lock.observed_mhz / hw["boost_clock_mhz"]
        )
    if probe:
        pct = 100 * probe["bw_read_gbs"] / hw["peak_mem_bw_gbs"]
        env |= probe | {"bw_read_pct_of_datasheet": pct}
    return env
