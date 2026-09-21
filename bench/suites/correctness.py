"""Correctness, judged before any speed claim: greedy tokens against the HF reference,
perplexity, GSM8K, and batch invariance, then the four gates on ours."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ..client.runner import run_bounded, run_open_loop
from ..gpu_tasks import run_task
from ..metrics.correctness import (
    batch_invariance,
    evaluate_gates,
    free_run_divergence,
    token_metrics,
)
from ..reference import ppl_windows, read_jsonl, write_jsonl
from ..run import Run
from ..workloads.workloads import Request
from .common import Session, datasets_dir, engine_session, guarded, load_datasets

GENERATION_CONCURRENCY = 16
GSM8K_CONCURRENCY = 32


def select_prompts(run: Run) -> list[dict]:
    """The first `n_prompts` ShareGPT prompts and the long prompts of the correctness set."""
    cfg = run.cfg.suite["correctness"]
    everything = read_jsonl(datasets_dir() / "correctness_prompts.jsonl")
    sharegpt = [p for p in everything if p["kind"] == "sharegpt"][: cfg["n_prompts"]]
    long = [p for p in everything if p["kind"] == "long"][: cfg["n_long_prompts"]]
    return sharegpt + long


def new_tokens(prompt: dict, cfg: dict) -> int:
    return cfg["long_max_new_tokens"] if prompt["kind"] == "long" else cfg["max_new_tokens"]


def reference_dir() -> Path:
    return datasets_dir() / "reference"


def ensure_reference(run: Run, prompts: list[dict]) -> None:
    """The reference's greedy continuations of `prompts`, generated once and cached."""
    cfg = run.cfg.suite["correctness"]
    run_task("reference-generate", model_path=run.cfg.model_path,
             prompts_path=str(datasets_dir() / "correctness_prompts.jsonl"),
             ids=[p["id"] for p in prompts], out_path=str(reference_dir() / "gen.jsonl"),
             new_tokens=cfg["max_new_tokens"], long_new_tokens=cfg["long_max_new_tokens"],
             dtype=run.cfg.model["dtype"])


def reference_ppl(run: Run) -> float:
    window = run.cfg.suite["correctness"]["ppl_window"]
    out = reference_dir() / f"ppl_w{window}.json"
    if not out.exists():
        bos = json.loads((datasets_dir() / "meta.json").read_text())["bos_token_id"]
        run_task("reference-ppl", model_path=run.cfg.model_path,
                 test_ids_path=str(datasets_dir() / "wikitext2_test_ids.npy"), window=window,
                 bos=bos, out_path=str(out), dtype=run.cfg.model["dtype"])
    return json.loads(out.read_text())["ppl"]


def generate_tokens(session: Session, prompts: list[dict], cfg: dict) -> list[dict]:
    """Greedy generations at concurrency 16: token ids and the engine's own logprobs."""
    requests = [Request(i, p["prompt_token_ids"], new_tokens(p, cfg))
                for i, p in enumerate(prompts)]
    records, _ = run_bounded(session.target(logprobs=True), requests, GENERATION_CONCURRENCY,
                             session.client_cpus)
    return [{"id": p["id"], "token_ids": r["token_ids"], "logprobs": r["token_logprobs"]}
            for p, r in zip(prompts, records, strict=True) if r["status"] == "ok"]


def run_batch_invariance(session: Session, prompts: list[dict], data) -> dict:
    """The same prompts alone (twice), amid filler requests, and amid filler in another order."""
    cfg = session.run.cfg.suite["correctness"]
    chosen = [p for p in prompts if p["kind"] == "sharegpt"][: cfg["batch_invariance_prompts"]]
    tokens = cfg["batch_invariance_tokens"]
    mine = [Request(i, p["prompt_token_ids"], tokens) for i, p in enumerate(chosen)]
    filler = [Request(len(mine) + i, rec["prompt_token_ids"], tokens)
              for i, rec in enumerate(data.sharegpt_pool[: cfg["batch_invariance_background"]])]

    def alone() -> list[list[int]]:
        out = []
        for req in mine:
            (rec,), _ = run_open_loop(session.target(logprobs=True), [req], [0.0],
                                      cpus=session.client_cpus)
            out.append(rec["token_ids"])
        return out

    def amid(order: list[Request]) -> list[list[int]]:
        records, _ = run_open_loop(session.target(logprobs=True), order, [0.0] * len(order),
                                   session.client_procs, session.client_cpus)
        by_id = {r["req_id"]: r["token_ids"] for r in records}
        return [by_id[req.req_id] for req in mine]

    rng = np.random.default_rng(session.run.cfg.suite["seed"])
    everything = [*mine, *filler]
    shuffled = [everything[i] for i in rng.permutation(len(everything))]
    return batch_invariance(alone(), alone(), amid(everything), amid(shuffled))


def measure_perplexity(session: Session) -> float | None:
    """exp of the mean negative logprob the engine assigns WikiText-2 test windows, or None
    when the engine cannot score a prompt."""
    cfg = session.run.cfg.suite["correctness"]
    data = load_datasets(session.run)
    test_ids = np.load(datasets_dir() / "wikitext2_test_ids.npy")
    total, count = 0.0, 0
    for window in ppl_windows(test_ids, cfg["ppl_window"], data.bos_token_id):
        logprobs = session.adapter.score(window)
        if logprobs is None:
            return None
        total, count = total + float(np.sum(logprobs)), count + len(logprobs)
    return float(np.exp(-total / count))


def run_gsm8k(session: Session, out_dir: Path) -> dict | None:
    """Five-shot GSM8K through `lm_eval`; None when the package is missing."""
    from ..env import package_versions

    if package_versions()["lm-eval"] is None:
        return None
    suite = session.run.cfg.suite["correctness"]
    args = (f"model={session.adapter.request_model},base_url={session.adapter.base_url()}"
            f"/v1/completions,tokenizer={session.run.cfg.model_path},tokenizer_backend=huggingface,"
            f"num_concurrent={GSM8K_CONCURRENCY},max_retries=3,tokenized_requests=False")
    seed = session.run.cfg.suite["seed"]
    command = [sys.executable, "-m", "lm_eval", "--model", "local-completions", "--tasks",
               "gsm8k", "--num_fewshot", str(suite["gsm8k_fewshot"]), "--seed", str(seed),
               "--model_args", args, "--output_path", str(out_dir)]
    if suite["gsm8k_limit"]:
        command += ["--limit", str(suite["gsm8k_limit"])]
    subprocess.run(command, check=True, capture_output=True, text=True)
    latest = max(out_dir.rglob("results_*.json"), key=lambda p: p.stat().st_mtime)
    results = json.loads(latest.read_text())["results"]["gsm8k"]
    return {"gsm8k_strict": results["exact_match,strict-match"],
            "gsm8k_flexible": results["exact_match,flexible-extract"]}


def measure_engine(run: Run, engine: str, prompts: list[dict], data) -> dict | None:
    """Everything that needs the engine up. None when its MUST checks abort it."""
    cfg = run.cfg.suite["correctness"]
    folder = run.phase_dir("correctness") / engine
    folder.mkdir(exist_ok=True)
    with guarded(run, "correctness", engine), engine_session(run, engine, "correctness") as s:
        gen = generate_tokens(s, prompts, cfg)
        write_jsonl(folder / "gen.jsonl", gen)
        out = {"batch_invariance": run_batch_invariance(s, prompts, data),
               "generated": len(gen), "requested": len(prompts)}
        if s.spec.real_model:
            out["ppl"] = measure_perplexity(s)
            out |= run_gsm8k(s, folder / "gsm8k") or {}
        return out
    return None


def judge_engine(run: Run, engine: str, prompts: list[dict], measured: dict) -> dict:
    """The reference's verdict on an engine's tokens, once the engine is down and the GPU free."""
    cfg = run.cfg.suite["correctness"]
    folder = run.dir / "correctness" / engine
    gen = read_jsonl(folder / "gen.jsonl")
    scored_path = folder / "scored.jsonl"
    run_task("reference-score", model_path=run.cfg.model_path,
             prompts_path=str(datasets_dir() / "correctness_prompts.jsonl"),
             ids=[p["id"] for p in prompts], engine_gen_path=str(folder / "gen.jsonl"),
             out_path=str(scored_path), dtype=run.cfg.model["dtype"])
    reference = read_jsonl(reference_dir() / "gen.jsonl")
    with_tokens = [g for g in gen if g["token_ids"] and len(g["logprobs"]) == len(g["token_ids"])]
    if not with_tokens:
        return measured | {"note": "the engine returned no token ids: token metrics unavailable"}
    scored = read_jsonl(scored_path)
    return (measured | token_metrics(with_tokens, scored, cfg["confident_gap_nats"])
            | free_run_divergence(with_tokens, reference) | {"ref_ppl": reference_ppl(run)})


def execute(run: Run, engines: list[str]) -> None:
    from ..config import load_engine

    prompts = select_prompts(run)
    data = load_datasets(run)
    real = [e for e in engines if load_engine(e, run.cfg.dir).real_model]
    if real:
        ensure_reference(run, prompts)
    summary: dict = {}
    for engine in engines:
        measured = measure_engine(run, engine, prompts, data)
        if measured is None:
            continue
        summary[engine] = (judge_engine(run, engine, prompts, measured) if engine in real
                           else measured)
    summary["gates"] = evaluate_gates(summary, run.cfg.suite["correctness"]["gates"])
    (run.dir / "correctness" / "summary.json").write_text(json.dumps(summary, indent=2))
    rows = [{"engine": e} | {k: v for k, v in s.items() if not isinstance(v, dict)}
            | {f"bi_{k}": v for k, v in s.get("batch_invariance", {}).items()}
            for e, s in summary.items() if e != "gates"]
    pd.DataFrame(rows).to_csv(run.dir / "tables" / "correctness.csv", index=False)
