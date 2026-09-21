"""`bench report`: one markdown report per run. Every number in it is read from a table under
`tables/` or from the correctness summary, never recomputed from memory."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import load_config
from .metrics.aggregate import aggregate_repeats
from .metrics.correctness import gates_verdict
from .plots import PLOTS, draw_all
from .plots.style import PlotContext
from .suites.common import datasets_dir

HEADLINE_PLOTS = ("p01", "p02", "p03", "p04", "p05")
SECTIONS = [
    ("Load results", ("p06", "p07", "p08", "p09")),
    ("Single-stream", ("p10", "p11", "p12")),
    ("Kernels", ("p13", "p14", "p15", "p16")),
    ("System overhead", ("p17", "p18")),
    ("Memory", ("p04", "p19", "p20")),
    ("Scheduler", ("p21", "p22")),
    ("Energy, cold start, soak", ("p23", "p25", "p24")),
]
BASELINES = ("vllm", "sglang", "trtllm")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def md_table(frame: pd.DataFrame, formats: dict[str, str] | None = None) -> str:
    """A GitHub markdown table; `formats` maps a column to a format spec."""
    formats = formats or {}

    def cell(column: str, value) -> str:
        if value is None or (isinstance(value, float) and value != value):
            return "n/a"
        if column in formats and isinstance(value, (int, float)):
            return format(value, formats[column])
        return str(value)

    head = "| " + " | ".join(frame.columns) + " |\n|" + "---|" * len(frame.columns) + "\n"
    rows = ["| " + " | ".join(cell(c, v) for c, v in row.items()) + " |"
            for row in frame.to_dict("records")]
    return head + "\n".join(rows) + "\n"


def load_tables(run_dir: Path) -> dict[str, pd.DataFrame]:
    tables = run_dir / "tables"
    return {p.stem: pd.read_csv(p) for p in tables.glob("*.csv") if p.stat().st_size}


def banners(env: dict, status: dict, checks: dict, summary: dict, tables: dict) -> list[str]:
    out = []
    gates = summary.get("gates", {})
    if any(g["passed"] is False for g in gates.values()):
        out.append("ENGINE FAILED CORRECTNESS: " + "; ".join(
            f"{k}: {g['detail']}" for k, g in gates.items() if g["passed"] is False))
    if not env.get("clocks_locked"):
        out.append(f"Clocks were not locked: {env.get('clock_lock_error') or 'not requested'}.")
    if env.get("quick"):
        out.append("Quick run: reduced repeats, durations and grids. Not for publication.")
    if read_json(datasets_dir() / "meta.json").get("synthetic"):
        out.append("Synthetic datasets: random tokens shaped like the real data.")
    points = tables.get("sweep_points")
    if points is not None and (~points.valid).any():
        out.append(f"{int((~points.valid).sum())} sweep point(s) stayed invalid or throttled "
                   f"after their rerun; they are drawn hollow.")
    for phase, entry in status.items():
        if entry["state"] in ("skipped", "failed", "aborted", "partial"):
            out.append(f"{phase}: {entry['state']}: {entry['note']}")
    for engine, c in checks.items():
        if not c.get("itl_valid", True):
            out.append(f"{engine}: ITL percentiles withheld (events are not one per token).")
        if not c.get("stats_available", True):
            out.append(f"{engine}: no stats endpoint; queue, KV and memory plots show n/a.")
        for r in c.get("results", []):
            if r["skipped"]:
                out.append(f"{engine}: check {r['name']} skipped: {r['detail']}")
    for gate, g in gates.items():
        if g["passed"] is None:
            out.append(f"Correctness gate {gate}: {g['detail']}")
    return out


def launch_commands(run_dir: Path) -> dict:
    """Each engine's resolved launch command and token budget, from its first sweep point."""
    out = {}
    for meta in sorted((run_dir / "sweep").glob("*/*/*.json")):
        data = json.loads(meta.read_text())
        out.setdefault(data["engine"], {"launch": data["launch"],
                                        "token_budget": data["token_budget"]})
    return out


def setup_block(env: dict, run_dir: Path, tuned: dict, suite: dict) -> str:
    gpu, model, hw = env.get("gpu", {}), env["model"], env["hardware"]
    versions = ", ".join(f"{k} {v}" for k, v in env.get("baselines", {}).items() if v)
    lines = [f"- GPU: {gpu.get('name', 'none')}, driver {gpu.get('driver', 'n/a')}, "
             f"CUDA runtime {env.get('cuda_runtime')}, torch {env.get('torch')}",
             f"- Engines: ours at commit {env.get('git_commit', '')[:7]}; "
             f"baselines {versions or 'none'}",
             f"- Model: {Path(model['path']).name}, dtype {model['dtype']}, "
             f"max_model_len {model['max_model_len']}, max_num_seqs {model['max_num_seqs']}, "
             f"gpu_mem_util {model['gpu_mem_util']}",
             f"- Clocks: SM {hw['gpu_clock_mhz']} MHz, memory {hw['mem_clock_mhz']} MHz, "
             f"locked: {env.get('clocks_locked')}",
             f"- Measured read bandwidth {env.get('bw_read_gbs', float('nan')):.0f} GB/s "
             f"({env.get('bw_read_pct_of_datasheet', float('nan')):.1f}% of the "
             f"{hw['peak_mem_bw_gbs']} GB/s datasheet)",
             "- SLOs: " + "; ".join(f"{w} TTFT {s['ttft_ms']} ms / TPOT {s['tpot_ms']} ms"
                                     for w, s in suite["slo"].items()),
             "- Tuned token budgets: " + (", ".join(f"{e} {t['token_budget']}"
                                                   for e, t in tuned.items()) or "none")]
    for engine, c in launch_commands(run_dir).items():
        lines.append(f"- Launch, {engine} (token budget {c['token_budget']}): "
                     f"`{' '.join(c['launch'])}`")
    return "\n".join(lines) + "\n"


def headline_table(tables: dict, summary: dict) -> pd.DataFrame:
    """One row per engine; each cell is read from a table, or n/a."""
    engines = sorted({e for t in tables.values() if "engine" in t for e in t.engine})
    capacity, sweep = tables.get("capacity"), tables.get("sweep_points")
    ref = capacity[capacity.workload == "sharegpt"].capacity_rps.max() if capacity is not None \
        else None
    rows = []
    for e in engines:
        row = {"engine": e}
        s = tables.get("sweep_summary")
        if s is not None:
            r = s[(s.engine == e) & (s.workload == "sharegpt")]
            row["max_sustainable_rps"] = r.max_sustainable_rps.iloc[0] if len(r) else None
        if capacity is not None:
            r = capacity[(capacity.engine == e) & (capacity.workload == "sharegpt")]
            row["capacity_tok_s"] = r.capacity_tok_s.iloc[0] if len(r) else None
        d = tables.get("single_decode")
        if d is not None:
            r = d[(d.engine == e) & (d.context == 512)]
            row["b1_tpot_ms"] = r.tpot_s.iloc[0] * 1000 if len(r) else None
            row["b1_bound_fraction"] = r.bound_fraction.iloc[0] if len(r) else None
        if sweep is not None and ref:
            mine = sweep[(sweep.engine == e) & (sweep.workload == "sharegpt") & sweep.valid]
            if len(mine):
                point = aggregate_repeats(
                    mine, ["engine", "offered_rps"], ["tpot_p99", "energy_j_per_out_token"])
                near = point.iloc[[(point.offered_rps - 0.5 * ref).abs().argmin()]]
                row["p99_tpot_ms_at_half_ref"] = near.tpot_p99.iloc[0] * 1000
                row["out_tok_per_joule_at_half_ref"] = (
                    1 / near.energy_j_per_out_token.iloc[0]
                    if near.energy_j_per_out_token.notna().iloc[0] else None)
        c = tables.get("coldstart")
        if c is not None:
            r = c[(c.engine == e) & (c.cache == "warm_cache")]
            row["cold_start_s"] = r.wait_ready_s.median() if len(r) else None
            row["first_request_penalty_ms"] = (r.first_request_penalty_s.median() * 1000
                                               if len(r) else None)
        rows.append(row)
    verdict = gates_verdict(summary["gates"]) if summary.get("gates") else "n/a"
    for row in rows:
        row["correctness"] = verdict if row["engine"] == "ours" else "n/a"
    return pd.DataFrame(rows)


HEADLINE_FORMATS = {"max_sustainable_rps": ".2f", "capacity_tok_s": ".0f", "b1_tpot_ms": ".2f",
                    "b1_bound_fraction": ".2f", "p99_tpot_ms_at_half_ref": ".1f",
                    "out_tok_per_joule_at_half_ref": ".3f", "cold_start_s": ".1f",
                    "first_request_penalty_ms": ".1f"}


def plot_lines(keys, drawn: dict) -> str:
    out = []
    for key in keys:
        name, title = PLOTS[key]
        out.append(f"![{title}](plots/{name}.png)\n" if drawn.get(key)
                   else f"*{title}: not run.*\n")
    return "\n".join(out)


def correctness_section(summary: dict) -> str:
    if not summary:
        return "*Correctness: not run.*\n"
    rows = [{"engine": e} | {k: v for k, v in s.items() if not isinstance(v, dict)}
            for e, s in summary.items() if e != "gates"]
    text = md_table(pd.DataFrame(rows), {c: ".5g" for c in pd.DataFrame(rows).columns})
    invariance = [{"engine": e} | s["batch_invariance"] for e, s in summary.items()
                  if e != "gates" and "batch_invariance" in s]
    if invariance:
        text += "\nBatch invariance:\n\n" + md_table(pd.DataFrame(invariance))
    gates = summary.get("gates", {})
    text += "\nGates on ours:\n\n" + "\n".join(
        f"- {k}: {'pass' if g['passed'] else 'fail' if g['passed'] is False else 'not evaluated'}"
        f" ({g['detail']})" for k, g in gates.items()) + "\n"
    return text


def method_notes(run_dir: Path, checks: dict) -> str:
    notes = ["Latency is measured by the client and includes HTTP and detokenization for every "
             "engine.",
             "Peak TFLOP/s and bandwidth are datasheet values; MFU is against the datasheet "
             "peak, which is quoted at the boost clock, above the locked clock.",
             "One model, one dtype, one GPU.",
             "The mock and null engines exist to test the harness and carry no results."]
    no_stats = [e for e, c in checks.items() if not c.get("stats_available", True)]
    if no_stats:
        notes.append("No stats endpoint for " + ", ".join(no_stats)
                     + ": their queue, KV utilisation and memory-breakdown values are n/a.")
    kernels = read_json(run_dir / "kernels" / "summary.json")
    if kernels.get("unavailable"):
        notes.append("Kernel contenders not run: " + "; ".join(
            f"{k}: {v}" for k, v in kernels["unavailable"].items()) + ".")
    if kernels.get("dropped_for_error"):
        notes.append("Kernel contenders dropped for numeric error above tolerance: "
                     + ", ".join(kernels["dropped_for_error"]) + ".")
    if kernels.get("skipped_decode_cells"):
        notes.append("Decode-attention cells skipped because their KV would not fit: "
                     + ", ".join(f"batch {b} x context {c}"
                                 for b, c in kernels["skipped_decode_cells"]) + ".")
    return "\n".join(f"- {n}" for n in notes) + "\n"


def build_report(run_dir: Path, config_dir: Path | None = None) -> tuple[Path, dict]:
    env = read_json(run_dir / "env.json")
    cfg = load_config(config_dir) if config_dir else load_config()
    suite = read_json_yaml(run_dir / "config_snapshot" / "suite.yaml") or cfg.suite
    status, checks = read_json(run_dir / "status.json"), read_json(run_dir / "checks.json")
    summary = read_json(run_dir / "correctness" / "summary.json")
    tuned = read_json(run_dir / "tune" / "tuned.json")
    tables = load_tables(run_dir)
    versions = {e: v for e, v in env.get("baselines", {}).items() if v}
    ctx = PlotContext(run_dir, run_dir / "plots", env, suite, versions,
                      correctness_failed=any(g["passed"] is False for g in
                                             summary.get("gates", {}).values()),
                      quick=bool(env.get("quick")))
    drawn = draw_all(ctx)
    headline = headline_table(tables, summary)
    parts = [f"# Benchmark report {run_dir.name}\n"]
    notices = banners(env, status, checks, summary, tables)
    if notices:
        parts.append("## Notices\n\n" + "\n".join(f"- {n}" for n in notices) + "\n")
    parts += ["## Setup\n\n" + setup_block(env, run_dir, tuned, suite),
              "## Headline table\n\n" + md_table(headline, HEADLINE_FORMATS),
              "## Headline plots\n\n" + plot_lines(HEADLINE_PLOTS, drawn),
              "## Correctness\n\n" + correctness_section(summary)]
    for title, keys in SECTIONS:
        body = plot_lines(keys, drawn)
        if title == "Memory" and "memory_exhaustion" in tables:
            body += "\nOverload behaviour:\n\n" + md_table(tables["memory_exhaustion"])
        parts.append(f"## {title}\n\n{body}")
        if title == "Scheduler" and "ablation" in tables:
            parts.append("## Ablation\n\n" + md_table(tables["ablation"]))
    parts.append("## Method notes\n\n" + method_notes(run_dir, checks))
    report = run_dir / "report.md"
    report.write_text("\n".join(parts))
    return report, {"headline": headline, "drawn": drawn, "summary": summary, "env": env,
                    "tables": tables, "notices": notices}


def read_json_yaml(path: Path) -> dict | None:
    import yaml

    return yaml.safe_load(path.read_text()) if path.exists() else None
