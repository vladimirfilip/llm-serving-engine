"""`python -m bench <command>`."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from . import env
from .config import CONFIG_DIR, load_config, load_engine
from .gpu_tasks import run_task
from .modelspec import ModelSpec
from .run import Run
from .suites import SUITE_ORDER, suite_function
from .suites.common import SuiteSkipped, datasets_dir, ensure_env
from .workloads.prepare import prepare_data
from .workloads.synthetic import write_synthetic

REQUIRED_HARDWARE = ("gpu_clock_mhz", "peak_tflops_bf16_dense", "peak_mem_bw_gbs")


def missing_hardware_fields(hw: dict) -> list[str]:
    return [key for key in REQUIRED_HARDWARE if hw.get(key) is None]


def cmd_env_check(args: argparse.Namespace) -> int:
    cfg = load_config()
    versions = env.package_versions()
    print(env.format_package_table(versions))
    missing = missing_hardware_fields(cfg.hardware)
    if missing and not args.allow_unlocked:
        print(f"hardware.yaml leaves required fields unset: {', '.join(missing)}", file=sys.stderr)
        return 1
    if not env.gpu_available():
        print("no NVIDIA driver: GPU checks skipped")
        return 0
    lock = env.lock_clocks(cfg.hardware)
    print(f"gpu: {json.dumps(env.gpu_facts(cfg.hardware['gpu_index']))}")
    print(f"clocks locked: {lock.locked} {lock.error}")
    if not lock.locked and not args.allow_unlocked:
        return 1
    return 0


def cmd_bwprobe(_args: argparse.Namespace) -> int:
    cfg = load_config()
    if missing := missing_hardware_fields(cfg.hardware):
        print(f"hardware.yaml leaves required fields unset: {', '.join(missing)}", file=sys.stderr)
        return 1
    lock = env.lock_clocks(cfg.hardware)
    probe = env.measure_bandwidth()
    pct = 100 * probe["bw_read_gbs"] / cfg.hardware["peak_mem_bw_gbs"]
    report = probe | {"clocks_locked": lock.locked, "bw_read_pct_of_datasheet": pct}
    print(json.dumps(report, indent=2))
    return 0


def add_run_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--engines", default="ours")
    parser.add_argument("--run-id")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-unlocked", action="store_true")
    parser.add_argument("--client-procs", type=int, default=1)
    parser.add_argument("--model-path", help="override model.yaml local_path")
    parser.add_argument("--config-dir", type=Path, help="read the YAML configs from here")


def open_run(args: argparse.Namespace) -> Run:
    cfg = load_config(args.config_dir or CONFIG_DIR, quick=args.quick)
    if args.model_path:
        cfg = dataclasses.replace(cfg, model=cfg.model | {"local_path": args.model_path})
    run = Run.open(cfg, args.run_id, force=args.force, allow_unlocked=args.allow_unlocked,
                   client_procs=args.client_procs)
    print(f"run {run.id}")
    return run


def run_phase(run: Run, name: str, engines: list[str]) -> bool:
    """Runs one suite unless it already finished; records the outcome. True on success."""
    if run.is_done(name):
        print(f"{name}: already done")
        return True
    run.set_status(name, "running")
    try:
        suite_function(name)(run, engines)
    except SuiteSkipped as e:
        run.mark_done(name)
        run.set_status(name, "skipped", str(e))
        print(f"{name}: skipped: {e}")
        return True
    except Exception as e:  # a failed phase is reported and must not stop later ones
        run.set_status(name, "failed", f"{type(e).__name__}: {e}")
        print(f"{name}: FAILED {type(e).__name__}: {e}", file=sys.stderr)
        return False
    if aborted := run.aborted(name):
        run.set_status(name, "partial", f"engines aborted: {', '.join(aborted)}")
        print(f"{name}: partial, aborted {aborted}", file=sys.stderr)
        return False
    run.mark_done(name)
    return True


def cmd_run(args: argparse.Namespace) -> int:
    return 0 if run_phase(open_run(args), args.suite, args.engines.split(",")) else 1


def cmd_tune(args: argparse.Namespace) -> int:
    return 0 if run_phase(open_run(args), "tune", args.engines.split(",")) else 1


def cmd_reference(args: argparse.Namespace) -> int:
    """Generate the reference continuations and perplexity, or with `--score ENGINE`
    teacher-force the reference over that engine's saved generations."""
    from .suites import correctness

    run = open_run(args)
    prompts = correctness.select_prompts(run)
    if args.score:
        folder = run.dir / "correctness" / args.score
        run_task("reference-score", model_path=run.cfg.model_path,
                 prompts_path=str(datasets_dir() / "correctness_prompts.jsonl"),
                 ids=[p["id"] for p in prompts], engine_gen_path=str(folder / "gen.jsonl"),
                 out_path=str(folder / "scored.jsonl"), dtype=run.cfg.model["dtype"])
    else:
        correctness.ensure_reference(run, prompts)
        print(f"reference perplexity {correctness.reference_ppl(run):.4f}")
    return 0


def find_run(args: argparse.Namespace) -> Path:
    from .run import RESULTS_DIR

    return RESULTS_DIR / args.run_id


def cmd_report(args: argparse.Namespace) -> int:
    from .report import build_report

    report, _ = build_report(find_run(args))
    print(report)
    return 0


def cmd_publish(args: argparse.Namespace) -> int:
    from .publish import MissingMarkers, publish

    try:
        unmet = publish(find_run(args), args.force, args.write_readme)
    except MissingMarkers as e:
        print(f"README not edited: {e}", file=sys.stderr)
        return 1
    if unmet and not args.force:
        print("not publishing; unmet conditions:\n- " + "\n- ".join(unmet), file=sys.stderr)
        return 1
    print("published to bench/published/")
    return 0


ALL_PHASES = ["tune", "correctness", "probe", "sweep", "single_stream", "kernels", "nsys",
              "memory", "scheduler", "ablation", "precision", "coldstart", "soak"]


def ensure_datasets(cfg, engines: list[str]) -> None:
    """Datasets are built once: from the real sources when any engine runs the real model, as
    seeded random tokens (and stamped so) when only fake engines are asked for."""
    if (datasets_dir() / "meta.json").exists():
        return
    real = any(load_engine(e, cfg.dir).real_model for e in engines)
    if real:
        prepare_data(cfg, datasets_dir())
    else:
        write_synthetic(datasets_dir(), cfg.suite,
                        ModelSpec.from_dir(cfg.model_path, cfg.model["dtype"]).vocab)


def cmd_all(args: argparse.Namespace) -> int:
    if args.quick and args.soak:
        print("--quick with --soak is an error", file=sys.stderr)
        return 2
    run = open_run(args)
    engines = args.engines.split(",")
    ensure_datasets(run.cfg, engines)
    ensure_env(run, any(load_engine(e, run.cfg.dir).uses_gpu for e in engines))
    ok = True
    for phase in ALL_PHASES:
        if phase == "soak" and not args.soak:
            continue
        ok &= run_phase(run, phase, engines)
    from .report import build_report

    print(build_report(run.dir)[0])
    return 0 if ok else 1


def cmd_prepare_data(args: argparse.Namespace) -> int:
    cfg = load_config()
    if args.model_path:
        cfg = dataclasses.replace(cfg, model=cfg.model | {"local_path": args.model_path})
    if args.synthetic:
        spec = ModelSpec.from_dir(cfg.model_path, cfg.model["dtype"])
        write_synthetic(datasets_dir(), cfg.suite, spec.vocab)
    else:
        prepare_data(cfg, datasets_dir())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bench")
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("env-check", help="package inventory, GPU idle and clock lock")
    check.add_argument("--allow-unlocked", action="store_true")
    check.set_defaults(fn=cmd_env_check)
    probe = commands.add_parser("bwprobe", help="measure read and copy bandwidth, no engine up")
    probe.set_defaults(fn=cmd_bwprobe)
    run = commands.add_parser("run", help="run one suite")
    run.add_argument("suite", choices=SUITE_ORDER)
    add_run_flags(run)
    run.set_defaults(fn=cmd_run)
    tune = commands.add_parser("tune", help="pick each engine's token budget")
    add_run_flags(tune)
    tune.set_defaults(fn=cmd_tune)
    ref = commands.add_parser("reference", help="HF reference generations, or score an engine")
    add_run_flags(ref)
    ref.add_argument("--score", metavar="ENGINE")
    ref.set_defaults(fn=cmd_reference)
    everything = commands.add_parser("all", help="every phase, then the report")
    add_run_flags(everything)
    everything.add_argument("--soak", action="store_true")
    everything.set_defaults(fn=cmd_all)
    report = commands.add_parser("report", help="write report.md for a run")
    report.add_argument("--run-id", required=True)
    report.set_defaults(fn=cmd_report)
    publish = commands.add_parser("publish", help="curate a run into bench/published/")
    publish.add_argument("--run-id", required=True)
    publish.add_argument("--write-readme", action="store_true")
    publish.add_argument("--force", action="store_true")
    publish.set_defaults(fn=cmd_publish)
    prep = commands.add_parser("prepare-data", help="build the seeded datasets")
    prep.add_argument("--synthetic", action="store_true",
                      help="random datasets for runs with no model files or network")
    prep.add_argument("--model-path")
    prep.set_defaults(fn=cmd_prepare_data)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)
