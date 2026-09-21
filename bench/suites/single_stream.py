"""Single-request behaviour through the server: decode speed against context, time to first
token against prompt length, and closed-loop throughput against batch size."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..client.runner import run_closed_loop, run_open_loop
from ..metrics.request_metrics import percentile
from ..modelspec import ModelSpec
from ..run import Run
from ..workloads.workloads import Request, fixed_requests
from .common import Session, engine_session, guarded, load_datasets, load_env

SATURATION_GAIN = 0.05
BATCH_PROMPT = 128
BATCH_OUTPUT = 512
WORKLOAD_STREAM = 100  # rng stream index: distinct from every load-sweep workload


def prompts(
    session: Session, data, length: int, output: int, n: int, repeat: int
) -> list[Request]:
    return fixed_requests(data.wikitext_train_ids, data.bos_token_id, length, output, n,
                          session.run.cfg.suite["seed"], WORKLOAD_STREAM + length % 97, repeat)


def one_at_a_time(session: Session, requests: list[Request]) -> list[dict]:
    records = []
    for req in requests:
        (rec,), _ = run_open_loop(session.target(), [req], [0.0], cpus=session.client_cpus)
        records.append(rec)
    return records


def decode_tpot(record: dict, skip: int, itl_valid: bool) -> float:
    """Mean gap after the first `skip` tokens; without one event per token, the whole decode."""
    times = record["token_times"]
    if itl_valid and len(times) > skip + 1:
        return (times[-1] - times[skip]) / (len(times) - 1 - skip)
    return (times[-1] - times[0]) / (record["completion_tokens_usage"] - 1)


def decode_versus_context(
    session: Session, data, spec: ModelSpec, bw_read_gbs: float | None
) -> list[dict]:
    cfg = session.run.cfg.suite["single_stream"]
    max_len = session.run.cfg.model["max_model_len"]
    rows = []
    for length in cfg["contexts"]:
        if length + cfg["decode_tokens"] > max_len:
            continue
        reqs = prompts(session, data, length, cfg["decode_tokens"], cfg["repeats"] + 1, 0)
        records = one_at_a_time(session, reqs)[1:]  # the first request is a warmup
        tpot = float(np.median([decode_tpot(r, cfg["skip_tokens"], session.adapter.itl_valid)
                                for r in records if r["status"] == "ok"]))
        mean_ctx = length + cfg["skip_tokens"] + (cfg["decode_tokens"] - cfg["skip_tokens"]) / 2
        bound = spec.bound_tok_s(mean_ctx, bw_read_gbs) if bw_read_gbs else None
        rows.append({"context": length, "tpot_s": tpot, "mean_ctx": mean_ctx,
                     "bound_tok_s": bound, "bound_fraction": (1 / tpot) / bound if bound else None})
    return rows


def ttft_versus_prompt(session: Session, data, spec: ModelSpec, peak_tflops: float) -> list[dict]:
    cfg = session.run.cfg.suite["single_stream"]
    max_len = session.run.cfg.model["max_model_len"]
    rows = []
    for length in cfg["ttft_prompts"]:
        if length + 2 > max_len:
            continue
        reqs = prompts(session, data, length, 2, 2 + cfg["ttft_repeats"], 1)
        records = one_at_a_time(session, reqs)[2:]  # two warmups
        ttft = float(np.median([r["token_times"][0] - r["t_send"] for r in records
                                if r["status"] == "ok"]))
        rows.append({"prompt": length, "ttft_s": ttft,
                     "prefill_mfu": spec.prefill_flops(length) / (ttft * peak_tflops * 1e12)})
    return rows


class FixedShapeSource:
    """Endless requests of one shape, each with its own random slice of the training text."""

    def __init__(self, data, length: int, output: int, seed: int):
        self.data, self.length, self.output = data, length, output
        self.rng = np.random.default_rng(seed)
        self.issued = 0

    def __call__(self, _worker: int) -> Request:
        start = int(self.rng.integers(0, len(self.data.wikitext_train_ids) - self.length))
        ids = self.data.wikitext_train_ids[start : start + self.length - 1].tolist()
        self.issued += 1
        return Request(self.issued, [self.data.bos_token_id, *ids], self.output)


def saturation_batch(rows: list[dict]) -> int | None:
    """The first batch size whose doubling gains under 5% output throughput."""
    by_batch = {r["batch"]: r["out_tok_s"] for r in rows}
    for batch in sorted(by_batch):
        if batch * 2 in by_batch and by_batch[batch * 2] < by_batch[batch] * (1 + SATURATION_GAIN):
            return batch
    return None


def throughput_versus_batch(session: Session, data) -> list[dict]:
    cfg = session.run.cfg.suite["single_stream"]
    rows = []
    for batch in cfg["batch_sizes"]:
        source = FixedShapeSource(data, BATCH_PROMPT, BATCH_OUTPUT, session.run.cfg.suite["seed"])
        records, _t0 = run_closed_loop(session.target(), source, batch, cfg["batch_total_s"],
                                       session.client_cpus)
        lo, hi = cfg["batch_warmup_s"], cfg["batch_total_s"]
        tokens = sum(int(np.count_nonzero((np.asarray(r["token_times"]) >= lo)
                                          & (np.asarray(r["token_times"]) <= hi)))
                     for r in records)
        tpots = [decode_tpot(r, 0, session.adapter.itl_valid) for r in records
                 if r["status"] == "ok" and r["t_first"] >= lo and r["completion_tokens_usage"] > 1]
        rows.append({"batch": batch, "out_tok_s": float(tokens) / (hi - lo),
                     "per_stream_tok_s": 1 / percentile(tpots, 50) if tpots else None})
    saturation = saturation_batch(rows)
    return [row | {"saturation_batch": saturation} for row in rows]


def execute(run: Run, engines: list[str]) -> None:
    data = load_datasets(run)
    env = load_env(run) if (run.dir / "env.json").exists() else {}
    tables: dict[str, list[dict]] = {"decode": [], "ttft": [], "batch": []}
    for engine in engines:
        with guarded(run, "single_stream", engine), engine_session(run, engine, "single") as s:
            env = load_env(run)
            spec = ModelSpec.from_dir(run.cfg.model_path, run.cfg.model["dtype"])
            peak = run.cfg.hardware["peak_tflops_bf16_dense"]
            for name, rows in (
                ("decode", decode_versus_context(s, data, spec, env.get("bw_read_gbs"))),
                ("ttft", ttft_versus_prompt(s, data, spec, peak)),
                ("batch", throughput_versus_batch(s, data)),
            ):
                tables[name] += [{"engine": engine} | row for row in rows]
    for name, rows in tables.items():
        pd.DataFrame(rows).to_csv(run.dir / "tables" / f"single_{name}.csv", index=False)
