"""The preconditions of a valid measurement. Every check runs once per run and engine; a
failing MUST check aborts that engine's run. Results go to `checks.json`."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass

import numpy as np

from ..client.runner import run_open_loop
from ..client.stream import Target
from ..engines.base import CheckFailed
from ..engines.server_proc import die_with_parent, free_port
from ..metrics.request_metrics import percentile, token_gaps
from ..workloads.arrivals import poisson_schedule
from ..workloads.workloads import Request
from .common import Session, load_datasets

PREFIX_RATIO_MIN = 0.8
TOKEN_ACCOUNTING_MIN = 0.995
BURST_GAP_FRACTION = 0.2
BURST_SHARE_MAX = 0.02
NULL_GAP_S = 0.010
IDLE_UTILIZATION_MAX_PCT = 5
IDLE_SETTLE_S = 30.0
NULL_TOKENS = 128
NULL_SERVER_WORKERS = 4
CLIENT_PROC_STEPS = (1, 2, 4, 8)


@dataclass(slots=True)
class CheckResult:
    name: str
    passed: bool
    detail: str
    must: bool = True
    skipped: bool = False


def checks_path(run):
    return run.dir / "checks.json"


def load_checks(run) -> dict:
    return json.loads(checks_path(run).read_text()) if checks_path(run).exists() else {}


def save_check_results(run, engine: str, results: list[CheckResult], itl_valid: bool,
                       client_procs: int) -> None:
    all_checks = load_checks(run)
    all_checks.setdefault(engine, {}).update(
        {"results": [asdict(r) for r in results], "itl_valid": itl_valid,
         "client_procs": client_procs})
    checks_path(run).write_text(json.dumps(all_checks, indent=2))


def _send(session: Session, requests: list[Request], t_sched: list[float], procs: int = 1,
          logprobs: bool = False) -> list[dict]:
    records, _ = run_open_loop(session.target(logprobs), requests, t_sched, procs,
                               session.client_cpus)
    return records


def _ttft(record: dict) -> float:
    return record["token_times"][0] - record["t_send"]


def check_gpu_idle(gpu_index: int, settle_s: float = IDLE_SETTLE_S) -> CheckResult:
    """Nothing else is using the GPU: no compute process and under 5% utilisation. A GPU that
    was busy a moment ago, with our own previous task, gets `settle_s` to go quiet."""
    from .. import env

    deadline = time.perf_counter() + settle_s
    while True:
        pids = env.compute_pids()
        utilization = env.gpu_utilization_pct(gpu_index)
        idle = not pids and utilization < IDLE_UTILIZATION_MAX_PCT
        if idle or time.perf_counter() >= deadline:
            return CheckResult("gpu_idle", idle,
                               f"compute processes {pids}, utilisation {utilization}%")
        time.sleep(1.0)


def require_idle_gpu(run, engine: str) -> None:
    """Before a GPU engine launches: records the idle check in `checks.json` and raises
    `CheckFailed` if the GPU is busy."""
    from .. import env

    if not env.gpu_available():
        return
    result = check_gpu_idle(run.cfg.hardware["gpu_index"])
    saved = load_checks(run)
    saved.setdefault(engine, {})["gpu_idle"] = asdict(result)
    checks_path(run).write_text(json.dumps(saved, indent=2))
    if not result.passed:
        raise CheckFailed(f"{engine}: GPU not idle before launch: {result.detail}")


def check_prefix_cache_off(session: Session) -> CheckResult:
    """Each of five distinct 4096-token prompts goes out twice; with no prefix cache the second
    first-token latency is about the first."""
    data = load_datasets(session.run)
    rng = np.random.default_rng([session.run.cfg.suite["seed"], 51])
    train = data.wikitext_train_ids
    ratios = []
    for _ in range(5):
        start = int(rng.integers(0, len(train) - 4096))
        prompt = [data.bos_token_id, *train[start : start + 4095].tolist()]
        first, second = (_send(session, [Request(0, prompt, 2)], [0.0])[0] for _ in range(2))
        ratios.append(_ttft(second) / _ttft(first))
    median = float(np.median(ratios))
    return CheckResult("prefix_cache_off", median >= PREFIX_RATIO_MIN,
                       f"median second/first TTFT {median:.2f} (need >= {PREFIX_RATIO_MIN})")


def natural_prompts(session: Session, n: int, length: int = 64) -> list[list[int]]:
    """WikiText slices, so a real model answers with text rather than byte fragments that a
    streamer legitimately holds back."""
    data = load_datasets(session.run)
    rng = np.random.default_rng([session.run.cfg.suite["seed"], 52])
    starts = rng.integers(0, len(data.wikitext_train_ids) - length, size=n)
    return [[data.bos_token_id, *data.wikitext_train_ids[s : s + length - 1].tolist()]
            for s in starts]


def check_token_accounting(session: Session) -> CheckResult:
    """One streamed event per token, so inter-token gaps mean something."""
    reqs = [Request(i, p, 256) for i, p in enumerate(natural_prompts(session, 20))]
    records = _send(session, reqs, [0.0] * 20)
    chunks = sum(r["n_chunks"] for r in records)
    tokens = sum(r["completion_tokens_usage"] for r in records)
    ratio = chunks / tokens if tokens else 0.0
    return CheckResult("token_accounting", ratio >= TOKEN_ACCOUNTING_MIN,
                       f"{chunks} events for {tokens} tokens ({ratio:.3f}); ITL "
                       f"{'valid' if ratio >= TOKEN_ACCOUNTING_MIN else 'invalid'}", must=False)


def check_no_bursts(session: Session) -> CheckResult:
    """Speculative decoding and token bundling both release tokens in bursts. Gaps are compared
    with the mean gap, not the median: once most gaps are burst gaps the median is one of them
    and nothing falls below a fraction of it."""
    record = _send(session, [Request(0, natural_prompts(session, 1)[0], 256)], [0.0])[0]
    gaps = token_gaps(record["token_times"])
    share = float((gaps < BURST_GAP_FRACTION * gaps.mean()).mean())
    return CheckResult("no_bursts", share < BURST_SHARE_MAX,
                       f"{share:.1%} of gaps under {BURST_GAP_FRACTION} of the mean (need < "
                       f"{BURST_SHARE_MAX:.0%})")


def check_tokenizer_identity(session: Session) -> CheckResult:
    from transformers import AutoTokenizer

    if not session.spec.has_tokenizer:
        return CheckResult("tokenizer_identity", True, "engine has no tokenizer", skipped=True)
    tokenizer = AutoTokenizer.from_pretrained(session.run.cfg.model_path)
    train = load_datasets(session.run).wikitext_train_ids
    worst = 0
    for i in range(20):
        text = tokenizer.decode(train[i * 200 : i * 200 + 12 + i].tolist())
        body = session.adapter.complete_text(text, 1)
        worst = max(worst, abs(body["usage"]["prompt_tokens"] - len(tokenizer.encode(text))))
    return CheckResult("tokenizer_identity", worst <= 1,
                       f"largest prompt_tokens difference {worst}")


def null_server_check(limits: dict, procs: int, cpus: str | None,
                      server_cpus: str | None = None) -> tuple[bool, str]:
    """The client against a server whose own timing is negligible: `limits["streams"]` concurrent
    10 ms streams must keep send lag and token-timestamp jitter under their p99 limits."""
    port = free_port()
    pin = ["taskset", "-c", server_cpus] if server_cpus else []
    server = subprocess.Popen([*pin, sys.executable, "-m", "bench.engines.null_server", "--port",
                               str(port), "--gap-ms", str(NULL_GAP_S * 1000), "--workers",
                               str(NULL_SERVER_WORKERS)], start_new_session=True,
                              preexec_fn=die_with_parent)
    try:
        time.sleep(1.0)
        rate = limits["streams"] / (NULL_TOKENS * NULL_GAP_S)
        n = 2 * limits["streams"]
        target = Target(f"http://127.0.0.1:{port}", "null", {}, True, timeout_s=60)
        requests = [Request(i, [1, 2, 3], NULL_TOKENS) for i in range(n)]
        schedule = poisson_schedule(rate, n, 0, 99, 0, 0).tolist()
        records, _ = run_open_loop(target, requests, schedule, procs, cpus)
    finally:
        os.killpg(server.pid, signal.SIGTERM)
        server.wait()
    lag = percentile([r["t_send"] - r["t_sched"] for r in records], 99)
    jitter = percentile(np.concatenate([np.abs(token_gaps(r["token_times"]) - NULL_GAP_S)
                                        for r in records if len(r["token_times"]) > 1]), 99)
    ok = lag < limits["send_lag_p99_ms"] / 1000 and jitter < limits["jitter_p99_ms"] / 1000
    return ok, (f"{procs} process(es): send_lag p99 {lag * 1e3:.2f} ms, "
                f"jitter p99 {jitter * 1e3:.2f} ms")


def check_client(session: Session) -> tuple[CheckResult, int]:
    """Climbs the client process counts until the null-server check passes; aborts past 8."""
    detail = []
    for procs in (p for p in CLIENT_PROC_STEPS if p >= session.client_procs):
        ok, message = null_server_check(session.run.cfg.suite["client_check"], procs,
                                        session.client_cpus, session.server_cpus)
        detail.append(message)
        if ok:
            return CheckResult("null_server_client", True, "; ".join(detail)), procs
    return CheckResult("null_server_client", False, "; ".join(detail)), CLIENT_PROC_STEPS[-1]


def ensure_checks(session: Session) -> None:
    """Runs every precondition check for this engine once per run, then applies the results: a
    failing MUST check raises, and ITL validity and the client process count carry forward."""
    run, engine = session.run, session.spec.name
    cached = load_checks(run).get(engine)
    if cached is None or "results" not in cached:
        client_result, procs = check_client(session)
        results = [
            check_prefix_cache_off(session), check_token_accounting(session),
            check_no_bursts(session), check_tokenizer_identity(session), client_result,
        ]
        itl_valid = all(r.passed for r in results if r.name in ("token_accounting", "no_bursts"))
        save_check_results(run, engine, results, itl_valid, procs)
        cached = load_checks(run)[engine]
    session.adapter.itl_valid = cached["itl_valid"]
    session.client_procs = max(session.client_procs, cached["client_procs"])
    failed = [r for r in cached["results"] if r["must"] and not r["passed"]]
    if failed:
        raise CheckFailed(f"{engine}: " + "; ".join(f"{r['name']}: {r['detail']}" for r in failed))

