"""A saved results directory with the tables and files each suite writes, for the report,
plot and publish tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ENGINES = ("vllm", "ours")
RATES = [1.0, 2.0, 3.0, 4.0]
WORKLOADS = ("sharegpt", "short_short", "long_short", "short_long")
HOME = "/home/alice"
HOST = "gpu-box-7"
MODEL_PATH = f"{HOME}/models/Llama-3.2-3B-Instruct"


def env_json(quick: bool = False, locked: bool = True, dirty: bool = False) -> dict:
    return {
        "git_commit": "abc1234deadbeef",
        "git_dirty": dirty,
        "clocks_locked": locked,
        "clock_lock_error": "",
        "hostname": HOST,
        "os": "Linux",
        "python": "3.10",
        "torch": "2.14",
        "cuda_runtime": "13.0",
        "quick": quick,
        "gpu": {"available": True, "name": "RTX 5070", "driver": "580"},
        "hardware": {
            "gpu_clock_mhz": 1900,
            "mem_clock_mhz": 14001,
            "peak_mem_bw_gbs": 672.0,
            "peak_tflops_bf16_dense": 61.75,
            "boost_clock_mhz": 2510,
        },
        "model": {
            "path": MODEL_PATH,
            "dtype": "bfloat16",
            "max_model_len": 16384,
            "max_num_seqs": 64,
            "gpu_mem_util": 0.9,
            "local_path": MODEL_PATH,
            "spec": {
                "d": 3072,
                "f": 8192,
                "n_layers": 28,
                "n_heads": 24,
                "n_kv_heads": 8,
                "head_dim": 128,
                "vocab": 128256,
                "dtype_bytes": 2,
            },
        },
        "bw_read_gbs": 637.0,
        "bw_copy_gbs": 580.0,
        "bw_read_pct_of_datasheet": 94.8,
        "baselines": {"vllm": "0.29.0"},
        "packages": {},
    }


def sweep_points(invalid: bool = False) -> pd.DataFrame:
    rows = []
    for engine, speed in (("vllm", 1.0), ("ours", 0.7)):
        for workload in WORKLOADS:
            for rate in RATES:
                for repeat in range(2):
                    over = rate * speed > 3
                    rows.append(
                        {
                            "engine": engine,
                            "workload": workload,
                            "offered_rps": rate,
                            "repeat": repeat,
                            "valid": not (invalid and engine == "ours" and rate == 4.0)
                    and not (invalid and engine == "ours" and rate == 2.0 and repeat == 1),
                            "throttled": False,
                            "attempts": 1,
                            "client_procs": 1,
                            "token_budget": 4096,
                            "invalid_reasons": "",
                            "achieved_rps": rate * (0.7 if over else 1),
                            "out_tok_s": 300 * rate * speed,
                            "slo_attainment": 0.5 if over else 1.0,
                            "goodput_rps": rate * 0.5,
                            "error_rate": 0.0,
                            "energy_j_per_out_token": 0.4 / speed,
                            **{
                                f"{m}_p{q}": (0.02 + 0.01 * rate + 0.002 * repeat) * (1 + q / 100)
                                for m in ("ttft", "tpot", "e2e", "itl")
                                for q in (50, 95, 99)
                            },
                        }
                    )
    return pd.DataFrame(rows)


def write(
    run_dir: Path,
    suites: set[str],
    quick: bool = False,
    locked: bool = True,
    dirty: bool = False,
    invalid: bool = False,
    gates_ok: bool = True,
) -> Path:
    tables = run_dir / "tables"
    for sub in (
        "tables",
        "plots",
        "sweep/ours/sharegpt",
        "correctness/ours",
        "scheduler/ours",
        "soak/ours",
        "nsys",
        "kernels",
        "tune",
        "probe",
        "config_snapshot",
    ):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    (run_dir / "env.json").write_text(json.dumps(env_json(quick, locked, dirty)))
    suite_yaml = Path(__file__).parents[1] / "configs" / "suite.yaml"
    (run_dir / "config_snapshot" / "suite.yaml").write_text(suite_yaml.read_text())
    checks = {e: {"itl_valid": True, "client_procs": 1, "results": []} for e in ENGINES}
    (run_dir / "checks.json").write_text(json.dumps(checks))
    status = {}
    (run_dir / "tune" / "tuned.json").write_text(
        json.dumps({e: {"token_budget": 4096, "candidates": {}} for e in ENGINES})
    )
    rng = np.random.default_rng(0)

    def csv(name: str, frame: pd.DataFrame) -> None:
        frame.to_csv(tables / f"{name}.csv", index=False)

    if "probe" in suites:
        csv(
            "capacity",
            pd.DataFrame(
                [
                    {"engine": e, "workload": w, "capacity_tok_s": 700 * s, "capacity_rps": 2.3 * s}
                    for e, s in (("vllm", 1.4), ("ours", 1.0))
                    for w in WORKLOADS
                ]
            ),
        )
    if "sweep" in suites:
        points = sweep_points(invalid)
        csv("sweep_points", points)
        csv(
            "sweep_summary",
            pd.DataFrame(
                [
                    {
                        "engine": e,
                        "workload": w,
                        "max_sustainable_rps": 2.0,
                        "n_points": 8,
                        "n_invalid": 0,
                    }
                    for e in ENGINES
                    for w in WORKLOADS
                ]
            ),
        )
        rate = 2.0
        (run_dir / "sweep" / "ours" / "sharegpt" / f"rate{rate:.3f}_rep0.json").write_text(
            json.dumps(
                {
                    "engine": "ours",
                    "launch": [
                        f"{HOME}/.bench-envs/ours/bin/python",
                        "-m",
                        "bench.engines.ours_server",
                        f"--model={MODEL_PATH}",
                    ],
                    "token_budget": 4096,
                    "offered_rps": rate,
                    "monitor": str(run_dir / "gpu.parquet"),
                    "t0": 0.0,
                    "window": [0, 1],
                }
            )
        )
        pd.DataFrame({"t": [0.5], "mem_used_bytes": [9 * 2**30]}).to_parquet(
            run_dir / "gpu.parquet"
        )
    if "single_stream" in suites:
        csv(
            "single_decode",
            pd.DataFrame(
                [
                    {
                        "engine": e,
                        "context": c,
                        "tpot_s": 0.012 * k + c * 2e-7,
                        "mean_ctx": c + 100,
                        "bound_tok_s": 90.0,
                        "bound_fraction": 0.85 / k,
                    }
                    for e, k in (("vllm", 1.0), ("ours", 1.3))
                    for c in (128, 512, 2048)
                ]
            ),
        )
        csv(
            "single_ttft",
            pd.DataFrame(
                [
                    {
                        "engine": e,
                        "prompt": p,
                        "ttft_s": 0.01 + p * 5e-5 * k,
                        "prefill_mfu": 0.4 / k,
                    }
                    for e, k in (("vllm", 1.0), ("ours", 1.5))
                    for p in (128, 512, 2048)
                ]
            ),
        )
        csv(
            "single_batch",
            pd.DataFrame(
                [
                    {
                        "engine": e,
                        "batch": b,
                        "out_tok_s": 90 * b / (1 + b / 40),
                        "per_stream_tok_s": 90 / (1 + b / 40),
                        "saturation_batch": 8,
                    }
                    for e in ENGINES
                    for b in (1, 2, 4, 8, 16)
                ]
            ),
        )
        status["single_stream"] = {"state": "done", "note": ""}
    if "ablation" in suites:
        csv(
            "ablation",
            pd.DataFrame(
                {
                    "step": ["naive_baseline", "+continuous_batching", "+paged_kv"],
                    "out_tok_s": [100.0, 300.0, 500.0],
                    "out_tok_s_min": [90.0, 280.0, 480.0],
                    "out_tok_s_max": [110.0, 320.0, 520.0],
                    "tpot_s": [0.05, 0.03, 0.02],
                    "tpot_s_min": [0.048, 0.029, 0.019],
                    "tpot_s_max": [0.052, 0.031, 0.021],
                }
            ),
        )
    if "memory" in suites:
        csv(
            "memory_breakdown",
            pd.DataFrame(
                [
                    {"engine": "ours", "phase": p, "component": c, "bytes": b}
                    for p in ("idle", "loaded")
                    for c, b in (("weights", 6 * 2**30), ("kv_cache", 3 * 2**30))
                ]
            ),
        )
        csv(
            "memory_device",
            pd.DataFrame(
                [
                    {
                        "engine": e,
                        "idle_bytes": 7 * 2**30,
                        "peak_bytes": 10 * 2**30,
                        "peak_source": "sweep",
                    }
                    for e in ENGINES
                ]
            ),
        )
        csv(
            "memory_grid",
            pd.DataFrame(
                [
                    {
                        "engine": e,
                        "length": L,
                        "batch": b,
                        "status": "ok" if b * L < 20000 else "failed",
                        "peak_running": b,
                        "preemptions": 0,
                    }
                    for e in ENGINES
                    for L in (512, 2048)
                    for b in (1, 8, 64)
                ]
            ),
        )
        csv(
            "memory_exhaustion",
            pd.DataFrame(
                [
                    {
                        "engine": e,
                        "outcome": "completed_all",
                        "recovered": True,
                        "oversubscription": 5.0,
                        "preemptions": 3,
                    }
                    for e in ENGINES
                ]
            ),
        )
        csv(
            "kv_util",
            pd.DataFrame(
                [
                    {"t_s": t, "kv_util": 0.5, "internal_frag": 0.1, "engine": e}
                    for e in ENGINES
                    for t in range(10)
                ]
            ),
        )
    if "correctness" in suites:
        gate = {"passed": gates_ok, "detail": "0.0005 <= 0.001"}
        summary = {
            "ours": {
                "abs_dlogprob_mean": 0.01,
                "tf_top1_agree": 0.99,
                "confident_mismatch_rate": 0.0005,
                "ppl": 7.0,
                "ref_ppl": 7.0,
                "batch_invariance": {"run_to_run_identical_fraction": 1.0},
            },
            "vllm": {"abs_dlogprob_mean": 0.01, "ppl": 7.0, "ref_ppl": 7.0},
            "gates": {
                "confident_mismatch_rate": gate,
                "abs_dlogprob_vs_baselines": {"passed": True, "detail": "ok"},
                "ppl_vs_reference": {"passed": True, "detail": "ok"},
                "gsm8k_vs_vllm": {"passed": True, "detail": "ok"},
            },
        }
        (run_dir / "correctness" / "summary.json").write_text(json.dumps(summary))
        for e in ENGINES:
            (run_dir / "correctness" / e).mkdir(exist_ok=True)
            gen = [{"id": "a", "token_ids": [1, 2, 3], "logprobs": [-1.0, -2.0, -3.0]}]
            scored = [{"id": "a", "ref_logprob": [-1.1, -2.0, -2.8]}]
            for name, records in (("gen", gen), ("scored", scored)):
                (run_dir / "correctness" / e / f"{name}.jsonl").write_text(
                    "".join(json.dumps(r) + "\n" for r in records)
                )
    if "kernels" in suites:
        csv(
            "kernels_decode",
            pd.DataFrame(
                [
                    {
                        "contender": "ours",
                        "kind": "decode",
                        "batch": b,
                        "ctx": c,
                        "gb_s": 30.0 * b,
                        "pct_of_bw_read": 5.0,
                        "ms_median": 0.3,
                        "dropped": False,
                        "max_abs_error": 1e-3,
                    }
                    for b in (1, 16, 128)
                    for c in (512, 2048, 8192)
                ]
            ),
        )
        csv(
            "kernels_prefill",
            pd.DataFrame(
                [
                    {
                        "contender": "ours",
                        "kind": "prefill",
                        "seq_len": n,
                        "tflop_s": 15.0,
                        "ms_median": 1.0,
                        "dropped": False,
                        "max_abs_error": 1e-3,
                    }
                    for n in (512, 2048, 8192)
                ]
            ),
        )
        gemm = [
            {
                "contender": c,
                "shape": s,
                "m": m,
                "n": 4096,
                "k": 3072,
                "tflop_s": 20.0,
                "gb_s": 300.0 * (1 if c == "ours" else 1.05),
                "ms_median": 0.1 * (1 + (c == "torch")),
                "ms_p10": 0.1,
                "ms_p90": 0.1,
            }
            for c in ("ours", "torch")
            for s in ("qkv_proj", "o_proj", "gate_up_proj", "down_proj", "lm_head")
            for m in (1, 16, 64, 256, 2048)
        ]
        csv("kernels_gemm", pd.DataFrame(gemm))
        (run_dir / "kernels" / "summary.json").write_text(
            json.dumps({"unavailable": {"flash_attn": "flash-attn is not installed"},
                    "dropped_for_error": ["flashinfer"],
                    "skipped_decode_cells": [[128, 8192]]})
        )
    if "nsys" in suites:
        csv(
            "nsys",
            pd.DataFrame(
                [
                    {
                        "engine": e,
                        "batch": b,
                        "gpu_busy_fraction": 0.8,
                        "gap_p50": 1e-5,
                        "gap_p99": 1e-3,
                        "gap_time_fraction_over_50us": 0.1,
                    }
                    for e in ENGINES
                    for b in (1, 16, 64)
                ]
            ),
        )
        pd.DataFrame(
            [
                {"engine": e, "batch": 16, "gap_s": g}
                for e in ENGINES
                for g in rng.uniform(1e-6, 1e-3, 50)
            ]
        ).to_parquet(run_dir / "nsys" / "gaps.parquet")
        csv(
            "nsys_steps",
            pd.DataFrame(
                [
                    {
                        "batch": b,
                        "period_s": 0.02,
                        "gpu_busy_s": 0.015,
                        "gpu_idle_s": 0.005,
                        "schedule_s": 0.001,
                        "prepare_inputs_s": 0.002,
                        "forward_s": 0.013,
                        "sample_s": 0.002,
                        "postprocess_s": 0.001,
                    }
                    for b in (1, 16, 64)
                ]
            ),
        )
    if "scheduler" in suites:
        gaps = pd.DataFrame(
            {
                "stream": 0,
                "t": np.linspace(0, 100, 200),
                "itl": rng.uniform(0.02, 0.05, 200),
                "label": ["baseline", "during_prefill"] * 100,
            }
        )
        gaps.to_parquet(run_dir / "scheduler" / "ours" / "interference_default.parquet")
        ov = pd.DataFrame(
            {
                "t_sched": np.linspace(0, 100, 100),
                "t_send": np.linspace(0, 100, 100),
                "t_first": np.linspace(0, 100, 100) + rng.uniform(1, 40, 100),
                "prompt_len": rng.integers(10, 1000, 100),
                "status": "ok",
            }
        )
        ov.to_parquet(run_dir / "scheduler" / "ours" / "overload.parquet")
        csv(
            "scheduler_interference",
            pd.DataFrame(
                [
                    {
                        "engine": "ours",
                        "variant": "default",
                        "label": "baseline",
                        "n": 100,
                        "itl_p50": 0.03,
                    }
                ]
            ),
        )
        csv(
            "scheduler_overload",
            pd.DataFrame([{"engine": "ours", "spearman_prompt_len_ttft": 0.4}]),
        )
    if "coldstart" in suites:
        csv(
            "coldstart",
            pd.DataFrame(
                [
                    {
                        "engine": e,
                        "launch": i,
                        "cache": c,
                        "wait_ready_s": 70.0,
                        "first_request_penalty_s": 0.05,
                    }
                    for e in ENGINES
                    for i, c in enumerate(("cold_cache", "warm_cache"))
                ]
            ),
        )
    if "soak" in suites:
        t = np.arange(0, 3600 * 4, 300.0)
        folder = run_dir / "soak" / "ours"
        pd.DataFrame({"t_s": t, "mem_used_bytes": 9e9, "server_rss_bytes": 3e9}).to_csv(
            folder / "memory.csv", index=False
        )
        pd.DataFrame({"t_s": t, "out_tok_s": 500.0}).to_csv(folder / "throughput.csv", index=False)
        pd.DataFrame({"t_s": t, "tpot_p99": 0.05}).to_csv(folder / "latency.csv", index=False)
        (folder / "verdict.json").write_text("{}")
    for phase in ("correctness", "probe", "sweep", "single_stream", "ablation"):
        if phase in suites:
            (run_dir / phase).mkdir(exist_ok=True)
            (run_dir / phase / "DONE").write_text("x")
            status[phase] = {"state": "done", "note": ""}
    (run_dir / "status.json").write_text(json.dumps(status))
    return run_dir


ALL_SUITES = {
    "probe",
    "sweep",
    "single_stream",
    "ablation",
    "memory",
    "correctness",
    "kernels",
    "nsys",
    "scheduler",
    "coldstart",
    "soak",
}
