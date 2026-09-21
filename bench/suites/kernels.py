"""Kernel microbenchmarks: ours against reference libraries, one layer at a time. Runs in a
child process (`bench.gpu_tasks`) so the harness never holds a CUDA context."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..gpu_tasks import run_task
from ..modelspec import ModelSpec
from ..run import Run
from .common import SuiteSkipped, ensure_env

DROP_NOTE = "max abs error against the fp32 reference above tolerance"


def time_closure(fn, warmup: int, iters: int, flush_mib: int) -> dict[str, float]:
    """Median, p10 and p90 of per-iteration CUDA-event times in ms. L2 is flushed before each
    timed iteration, outside the timed region."""
    import torch

    scratch = torch.empty(flush_mib * 2**20 // 4, dtype=torch.float32, device="cuda")
    for _ in range(warmup):
        fn()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for start, end in zip(starts, ends, strict=True):
        scratch.zero_()
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    ms = np.array([s.elapsed_time(e) for s, e in zip(starts, ends, strict=True)])
    return {"ms_median": float(np.median(ms)), "ms_p10": float(np.percentile(ms, 10)),
            "ms_p90": float(np.percentile(ms, 90))}


def attention_rows(contenders, kind: str, points: list[tuple], spec: ModelSpec, cfg: dict,
                   bw_read_gbs: float | None, failures: dict[str, str]) -> list[dict]:
    """One row per (contender, point). `points` are (batch, ctx) for decode, (seq_len,) for
    prefill. A contender above the numeric tolerance is kept as a dropped row; one that raises
    is recorded in `failures` with the reason and skipped from then on."""
    from ..kernels.inputs import ATTENTION_TOLERANCE

    rows = []
    for adapter in contenders:
        for point in points:
            if adapter.name in failures:
                break
            build, error = ((adapter.decode_attention, adapter.decode_error) if kind == "decode"
                            else (adapter.prefill_attention, adapter.prefill_error))
            try:
                fn = build(*point, spec)
                if fn is None:
                    continue
                err = error(*point, spec)
                if err is None or err <= ATTENTION_TOLERANCE:
                    t = time_closure(fn, cfg["warmup_iters"], cfg["timed_iters"],
                                     cfg["l2_flush_mib"])
            except Exception as e:  # a contender that cannot run here is reported by name
                failures[adapter.name] = f"{type(e).__name__}: {str(e).strip()[-300:]}"
                break
            row = {"contender": adapter.name, "kind": kind, "max_abs_error": err,
                   "dropped": err is not None and err > ATTENTION_TOLERANCE}
            row |= dict(zip(("batch", "ctx") if kind == "decode" else ("seq_len",), point,
                            strict=True))
            if not row["dropped"]:
                seconds = t["ms_median"] / 1000
                if kind == "decode":
                    moved = point[0] * point[1] * spec.kv_bytes_per_tok_layer
                    row["gb_s"] = moved / seconds / 1e9
                    row["pct_of_bw_read"] = 100 * row["gb_s"] / bw_read_gbs if bw_read_gbs else None
                else:
                    flops = 2 * point[0] ** 2 * spec.n_heads * spec.head_dim
                    row["tflop_s"] = flops / seconds / 1e12
                row |= t
            rows.append(row)
    return rows


def gemm_rows(contenders, spec: ModelSpec, cfg: dict, failures: dict[str, str]) -> list[dict]:
    """Every contender's time per projection and M, and `ratio_to_torch` (ours over torch time)."""
    from ..kernels.protocol import GEMM_SHAPES, gemm_shape

    rows = []
    for which in GEMM_SHAPES:
        n, k = gemm_shape(which, spec)
        for m in cfg["gemm_m"]:
            for adapter in contenders:
                if adapter.name in failures:
                    continue
                try:
                    fn = adapter.gemm(which, m, spec)
                    if fn is None:
                        continue
                    t = time_closure(fn, cfg["warmup_iters"], cfg["timed_iters"],
                                     cfg["l2_flush_mib"])
                except Exception as e:  # a contender that cannot run here is reported by name
                    failures[adapter.name] = f"{type(e).__name__}: {str(e).strip()[-300:]}"
                    continue
                seconds = t["ms_median"] / 1000
                moved = (n * k + m * (n + k)) * spec.dtype_bytes
                rows.append({"contender": adapter.name, "shape": which, "m": m, "n": n, "k": k,
                             "tflop_s": 2 * m * n * k / seconds / 1e12,
                             "gb_s": moved / seconds / 1e9} | t)
    torch_ms = {(r["shape"], r["m"]): r["ms_median"] for r in rows if r["contender"] == "torch"}
    for r in rows:
        r["ratio_to_torch"] = r["ms_median"] / torch_ms[(r["shape"], r["m"])] \
            if (r["shape"], r["m"]) in torch_ms else None
    return rows


def task_kernels(model_path: str, dtype: str, cfg: dict, bw_read_gbs: float | None,
                 out_dir: str) -> dict:
    """Run every kernel benchmark and write the tables; returns the contenders that ran and the
    ones that could not."""
    import torch

    from ..kernels.ours_kernels import OursKernels
    from ..kernels.reference_kernels import available_contenders

    spec = ModelSpec.from_dir(model_path, dtype)
    references, missing = available_contenders()
    contenders = [OursKernels(), *references]
    total = torch.cuda.get_device_properties(0).total_memory
    budget = cfg["max_kv_fraction_of_gpu_mem"] * total
    grid = [(b, c) for b in cfg["decode_batches"] for c in cfg["decode_contexts"]]
    decode_points = [(b, c) for b, c in grid if b * c * spec.kv_bytes_per_token <= budget]
    skipped_cells = [cell for cell in grid if cell not in decode_points]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    failures: dict[str, str] = {}
    tables = {
        "kernels_decode": attention_rows(contenders, "decode", decode_points, spec, cfg,
                                         bw_read_gbs, failures),
        "kernels_prefill": attention_rows(contenders, "prefill",
                                          [(n,) for n in cfg["prefill_lengths"]], spec, cfg,
                                          bw_read_gbs, failures),
        "kernels_gemm": gemm_rows(contenders, spec, cfg, failures),
    }
    for name, rows in tables.items():
        pd.DataFrame(rows).to_csv(out / f"{name}.csv", index=False)
    dropped = sorted({r["contender"] for rows in tables.values() for r in rows if r.get("dropped")})
    return {"contenders": {c.name: c.notes for c in contenders},
            "unavailable": missing | {f"{n} (failed at run time)": r for n, r in failures.items()},
            "dropped_for_error": dropped, "skipped_decode_cells": skipped_cells,
            "gpu": torch.cuda.get_device_name(0)}


def execute(run: Run, engines: list[str]) -> None:
    from .. import env

    if "ours" not in engines:
        raise SuiteSkipped("kernels compare ours against reference libraries")
    if not env.gpu_available():
        raise SuiteSkipped("no NVIDIA driver")
    bw = ensure_env(run).get("bw_read_gbs")
    summary = run_task("kernels", model_path=run.cfg.model_path, dtype=run.cfg.model["dtype"],
                       cfg=run.cfg.suite["kernels"], bw_read_gbs=bw,
                       out_dir=str(run.dir / "tables"))
    (run.phase_dir("kernels") / "summary.json").write_text(json.dumps(summary, indent=2))
