"""Loads the YAML configs. Every number the harness needs comes from here or from the model's
`config.json`; nothing else hard-codes dimensions, paths, GPU names or SLOs."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).parent / "configs"
REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True, slots=True)
class Config:
    hardware: dict
    model: dict
    suite: dict
    quick: bool = False
    dir: Path = CONFIG_DIR

    @property
    def model_path(self) -> str:
        return str(Path(self.model["local_path"]).expanduser())

    @property
    def workloads(self) -> dict[str, dict]:
        """Workloads this model length can run. A workload naming `requires_max_model_len`
        disables itself when the engines are launched with less."""
        return {
            name: w for name, w in self.suite["workloads"].items()
            if self.model["max_model_len"] >= w.get("requires_max_model_len", 0)
        }

    def workload_index(self, name: str) -> int:
        return list(self.suite["workloads"]).index(name)


@dataclass(frozen=True, slots=True)
class EngineSpec:
    name: str
    python: str | None
    launch: list[str]
    package: str | None = None  # distribution whose version identifies this engine
    accepts_token_ids: bool = True
    uses_gpu: bool = True
    has_tokenizer: bool = True
    real_model: bool = True  # False for a fake engine, which has no reference to be judged against
    request_model: str = "{served_model_name}"
    extra_body: dict = field(default_factory=dict)
    health_path: str = "/health"
    env: dict = field(default_factory=dict)
    path_prepend: list[str] = field(default_factory=list)  # directories put ahead of PATH
    tuning: dict = field(default_factory=dict)
    metrics: dict | None = None
    stats_api: str = "none"  # "internal" (/internal/stats), "prometheus" or "none"
    score_api: str = "none"  # "internal", "vllm", "sglang" or "none"
    variants: dict = field(default_factory=dict)  # name -> {args_add, env} launch differences
    files: dict = field(default_factory=dict)

    @property
    def token_budgets(self) -> list[int | None]:
        return self.tuning.get("token_budget") or [None]

    @property
    def default_token_budget(self) -> int | None:
        """The budget for a run that skipped `bench tune`."""
        return self.tuning.get("default", self.token_budgets[0])


def load_config(config_dir: Path = CONFIG_DIR, quick: bool = False) -> Config:
    def read(name: str) -> dict:
        return yaml.safe_load((config_dir / name).read_text())

    suite = read("suite.yaml")
    return Config(read("hardware.yaml"), read("model.yaml"), apply_quick(suite) if quick else suite,
                  quick, config_dir)


def load_engine(name: str, config_dir: Path = CONFIG_DIR) -> EngineSpec:
    return EngineSpec(**yaml.safe_load((config_dir / "engines" / f"{name}.yaml").read_text()))


def apply_quick(suite: dict) -> dict:
    """The `--quick` overrides: fewer repeats, shorter windows, a coarser grid."""
    s = copy.deepcopy(suite)
    sweep = s["load_sweep"]
    sweep["target_duration_s"] = 30
    sweep["probe_requests"] = 128
    for w in sweep["per_workload"].values():
        w["repeats"] = 1
        w["fractions"] = w["fractions"][::2]
    ss = s["single_stream"]
    ss.update(repeats=2, ttft_repeats=3, batch_total_s=25, batch_warmup_s=5)
    s["memory"].update(grid_batches=[1, 8, 64], grid_lengths=[512, 4096], kv_util_duration_s=60)
    s["correctness"].update(n_prompts=20, gsm8k_limit=100)
    return s


def expand(template: list[str] | str, values: dict) -> list[str] | str:
    """Fills `{name}` fields and expands a leading `~`, bare or after `=`, in every argument."""
    def one(text: str) -> str:
        return re.sub(r"(^|=)~(?=/|$)", lambda m: m.group(1) + str(Path.home()),
                      text.format_map(values))

    return one(template) if isinstance(template, str) else [one(t) for t in template]
