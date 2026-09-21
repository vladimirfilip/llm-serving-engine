"""`bench publish`: curated, sanitised output of one run, committed under `bench/published/`,
and optionally the benchmark block of the root README."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pandas as pd

from .config import BENCH_DIR, REPO_ROOT
from .plots import PLOTS
from .report import HEADLINE_FORMATS, HEADLINE_PLOTS, build_report, md_table

PUBLISHED = BENCH_DIR / "published"
START, END = "<!-- BENCH:START -->", "<!-- BENCH:END -->"
REQUIRED_PHASES = ("correctness", "probe", "sweep", "single_stream", "ablation")


class MissingMarkers(Exception):
    """The root README has no benchmark block to replace."""


def unmet_conditions(run_dir: Path) -> list[str]:
    """Everything that makes this run unfit to publish."""
    env = json.loads((run_dir / "env.json").read_text())
    summary_path = run_dir / "correctness" / "summary.json"
    gates = json.loads(summary_path.read_text()).get("gates", {}) if summary_path.exists() else {}
    unmet = []
    if env.get("quick"):
        unmet.append("the run was --quick")
    if not env.get("clocks_locked"):
        unmet.append("clocks were not locked")
    if len(gates) < 4 or any(g["passed"] is None for g in gates.values()):
        unmet.append("not all four correctness gates were evaluated")
    elif not all(g["passed"] for g in gates.values()):
        unmet.append("a correctness gate failed")
    if env.get("git_dirty"):
        unmet.append("the engine tree was dirty during the run")
    points = run_dir / "tables" / "sweep_points.csv"
    if points.exists():
        table = pd.read_csv(points)
        bad = table[(table.workload == "sharegpt") & (~table.valid | table.throttled.fillna(False))]
        if len(bad):
            unmet.append(f"{len(bad)} sharegpt sweep point(s) still invalid or throttled")
    unfinished = [p for p in REQUIRED_PHASES if not (run_dir / p / "DONE").exists()]
    if unfinished:
        unmet.append("phases not finished: " + ", ".join(unfinished))
    return unmet


def sanitizer(env: dict):
    """A function removing usernames, hostnames and absolute paths from text."""
    model = env["model"]["path"]
    pairs = [(model, Path(model).name), (env.get("hostname", ""), "<host>"),
             (str(Path.home()), "~"), (str(REPO_ROOT), ".")]

    def clean(text: str) -> str:
        for old, new in pairs:
            if old:
                text = text.replace(old, new)
        text = re.sub(r"/home/[^/\s\"']+", "~", text)
        return re.sub(r"(?<![\w.])/root\b", "~", text)

    return clean


def commands(run_dir: Path) -> dict:
    out = {}
    for meta in sorted((run_dir / "sweep").glob("*/*/*.json")):
        data = json.loads(meta.read_text())
        out.setdefault(data["engine"], {"launch": data["launch"],
                                        "token_budget": data["token_budget"]})
    return out


def verdict(gate: dict) -> str:
    return "not evaluated" if gate["passed"] is None else "pass" if gate["passed"] else "fail"


def readme_block(run_dir: Path, info: dict, unmet: list[str], forced: bool) -> str:
    env, headline = info["env"], info["headline"]
    gpu, model = env["gpu"]["name"], Path(env["model"]["path"]).name
    versions = ", ".join(f"{k} {v}" for k, v in env.get("baselines", {}).items() if v)
    date = run_dir.name.split("-")[0]
    gates = info["summary"].get("gates", {})
    lines = [
        START, "## Benchmarks", "",
        f"{model} in {env['model']['dtype']} on a {gpu}, run {date[:4]}-{date[4:6]}-{date[6:]}, "
        f"engine commit {env['git_commit'][:7]}. Baselines: {versions or 'none'}.", "",
    ]
    if forced:
        lines += [f"Published with --force. Unmet conditions: {'; '.join(unmet)}.", ""]
    lines += [
        "### Scope", "",
        "- Same checkpoint, dtype, GPU and locked clocks for every engine.",
        "- Prefix caching and speculative decoding are disabled on the baselines because this "
        "engine does not implement them.",
        "- Baselines are tuned with the same procedure as this engine.",
        "- One model only. Client-side timing includes HTTP and detokenization.",
        *[f"- Correctness gate {k}: {verdict(g)} ({g['detail']})" for k, g in gates.items()], "",
        "### Results", "", md_table(headline, HEADLINE_FORMATS)]
    for key in HEADLINE_PLOTS:
        name, title = PLOTS[key]
        lines += [f"![{title}](bench/published/plots/{name}.png)", f"*{title}.*", ""]
    lines += ["### Reproduce", "",
              "```", "python -m bench prepare-data", "python -m bench all --engines "
              "ours,vllm,sglang,trtllm", f"python -m bench publish --run-id {run_dir.name} "
              "--write-readme", "```", "",
              "See [bench/published/report.md](bench/published/report.md) and "
              "[bench/README.md](bench/README.md).", END]
    return "\n".join(lines) + "\n"


def replace_block(readme: str, block: str) -> str:
    if START not in readme or END not in readme or readme.index(START) > readme.index(END):
        raise MissingMarkers(f"add {START} and {END} to the README where the results belong")
    head, rest = readme.split(START, 1)
    tail = rest.split(END, 1)[1]
    return head + block.rstrip("\n") + tail


def publish(run_dir: Path, force: bool = False, write_readme: bool = False,
            published: Path = PUBLISHED, readme: Path = REPO_ROOT / "README.md") -> list[str]:
    """Returns the unmet conditions (empty on a clean publish). Without `force` nothing is
    written when any is unmet."""
    unmet = unmet_conditions(run_dir)
    if unmet and not force:
        return unmet
    _, info = build_report(run_dir)
    env = info["env"]
    clean = sanitizer(env)
    if published.exists():
        shutil.rmtree(published)
    (published / "plots").mkdir(parents=True)
    (published / "tables").mkdir()
    for png in (run_dir / "plots").glob("*.png"):
        shutil.copy2(png, published / "plots" / png.name)
    for csv in (run_dir / "tables").glob("*.csv"):
        (published / "tables" / csv.name).write_text(clean(csv.read_text()))
    (published / "report.md").write_text(clean((run_dir / "report.md").read_text()))
    env_public = {k: v for k, v in env.items() if not k.startswith("_")}
    (published / "env.json").write_text(clean(json.dumps(env_public, indent=2)))
    (published / "commands.json").write_text(clean(json.dumps(commands(run_dir), indent=2)))
    (published / "run.json").write_text(clean(json.dumps({
        "run_id": run_dir.name, "engine_commit": env["git_commit"], "date": run_dir.name[:8],
        "model": Path(env["model"]["path"]).name, "gpu": env["gpu"]["name"],
        "dtype": env["model"]["dtype"], "baselines": env.get("baselines", {})}, indent=2)))
    block = clean(readme_block(run_dir, info, unmet, force and bool(unmet)))
    (published / "readme_section.md").write_text(block)
    if write_readme:
        readme.write_text(replace_block(readme.read_text(), block))
    return unmet if force else []
