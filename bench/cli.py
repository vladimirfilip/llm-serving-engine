"""`python -m bench <command>`."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys

from . import env
from .config import load_config
from .gpu_tasks import run_task
from .modelspec import ModelSpec
from .run import Run
from .suites import SUITE_ORDER, suite_function
from .suites.common import SuiteSkipped, datasets_dir
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


def open_run(args: argparse.Namespace) -> Run:
    cfg = load_config(quick=args.quick)
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
    prep = commands.add_parser("prepare-data", help="build the seeded datasets")
    prep.add_argument("--synthetic", action="store_true",
                      help="random datasets for runs with no model files or network")
    prep.add_argument("--model-path")
    prep.set_defaults(fn=cmd_prepare_data)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)
