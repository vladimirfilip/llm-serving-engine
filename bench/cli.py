"""`python -m bench <command>`."""

from __future__ import annotations

import argparse
import json
import sys

from . import env
from .config import load_config

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
    probe = env.bandwidth_probe()
    pct = 100 * probe["bw_read_gbs"] / cfg.hardware["peak_mem_bw_gbs"]
    print(json.dumps(probe | {"clocks_locked": lock.locked, "bw_read_pct_of_datasheet": pct}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bench")
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("env-check", help="package inventory, GPU idle and clock lock")
    check.add_argument("--allow-unlocked", action="store_true")
    check.set_defaults(fn=cmd_env_check)
    probe = commands.add_parser("bwprobe", help="measure read and copy bandwidth, no engine up")
    probe.set_defaults(fn=cmd_bwprobe)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)
