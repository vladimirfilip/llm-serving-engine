import asyncio
import sys
import time

import numpy as np
import pytest

from bench.client.runner import closed_loop, run_open_loop
from bench.client.stream import Target, new_session, send
from bench.config import load_config, load_engine
from bench.engines.base import Launch
from bench.engines.generic import make_adapter
from bench.run import Run
from bench.workloads.workloads import Request

PREFILL_MS_PER_1K = 100.0
DECODE_MS = 20.0


@pytest.fixture(scope="module")
def mock_adapter(tmp_path_factory):
    cfg = load_config()
    run = Run.open(cfg, "t", results_dir=tmp_path_factory.mktemp("results"))
    spec = load_engine("mock")
    adapter = make_adapter(spec, cfg, run)
    adapter.launch(Launch(args_add=[f"--prefill-ms-per-1k={PREFILL_MS_PER_1K}",
                                    f"--decode-ms={DECODE_MS}"]), "test")
    adapter.wait_ready(timeout_s=30)
    yield adapter
    adapter.shutdown()


def target_for(adapter) -> Target:
    return Target(adapter.base_url(), adapter.request_model, adapter.extra_body,
                  adapter.accepts_token_ids, timeout_s=30, logprobs=True)


def one_request(adapter, prompt_len: int, output_len: int) -> dict:
    req = Request(0, [1, *range(2, prompt_len + 1)], output_len)
    records, _ = run_open_loop(target_for(adapter), [req], [0.0])
    return records[0]


def test_tpot_and_ttft_match_the_mock_within_the_spec_tolerances(mock_adapter):
    """TPOT is absolute. TTFT is a difference between a long and a short prompt in one
    session: the fixed HTTP round trip (about 2 ms on a loaded VM) belongs to the machine,
    and the difference isolates the prefill time the client is meant to measure."""
    lengths = [16, 16, 16, 1016, 16, 1016, 16, 1016]
    reqs = [Request(i, [1, *range(2, n + 1)], 10) for i, n in enumerate(lengths)]
    records, _ = run_open_loop(target_for(mock_adapter), reqs, [0.4 * i for i in range(8)])
    assert all(r["status"] == "ok" and r["n_chunks"] == 10 == r["completion_tokens_usage"]
               for r in records)
    ttft = [r["token_times"][0] - r["t_send"] for r in records]
    short, long = np.median([ttft[i] for i in (2, 4, 6)]), np.median([ttft[i] for i in (3, 5, 7)])
    assert abs((long - short) - PREFILL_MS_PER_1K / 1000) < 0.002
    times = np.array(records[-1]["token_times"])
    assert abs((times[-1] - times[0]) / 9 - DECODE_MS / 1000) < 0.001
    assert records[-1]["prompt_tokens_usage"] == 1016 and len(records[-1]["token_logprobs"]) == 10


def test_open_loop_sends_at_the_scheduled_time_even_when_earlier_requests_are_slow(mock_adapter):
    reqs = [Request(i, [1, 2, 3], 40) for i in range(6)]
    schedule = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
    records, _ = run_open_loop(target_for(mock_adapter), reqs, schedule)
    lags = [r["t_send"] - r["t_sched"] for r in records]
    assert max(lags) < 0.005
    assert min(r["t_done"] for r in records) > max(schedule)  # every request outlived the schedule


def test_sharding_across_processes_keeps_every_request_and_the_shared_clock(mock_adapter):
    reqs = [Request(i, [1, 2, 3], 5) for i in range(8)]
    schedule = [0.1 * i for i in range(8)]
    records, t0 = run_open_loop(target_for(mock_adapter), reqs, schedule, procs=2)
    assert [r["req_id"] for r in records] == list(range(8))
    assert all(r["status"] == "ok" for r in records)
    assert all(abs(r["t_send"] - r["t_sched"]) < 0.02 for r in records)
    assert t0 > 0


def test_closed_loop_keeps_concurrency_and_stops_issuing_at_the_deadline(mock_adapter):
    def next_request(_worker: int) -> Request:
        return Request(0, [1, 2, 3], 5)

    async def go():
        target = target_for(mock_adapter)
        return await closed_loop(target, next_request, 3, 1.0, time.perf_counter())

    records = asyncio.run(go())
    per_request = 0.02 * 5
    assert 3 * (1.0 / per_request) * 0.6 < len(records) < 3 * (1.0 / per_request) * 1.1
    assert max(r["t_send"] for r in records) < 1.0 + 0.05


def test_a_failing_server_is_recorded_as_an_error_not_raised(mock_adapter):
    dead = Target("http://127.0.0.1:1", "m", {}, True, timeout_s=2)

    async def go():
        async with new_session() as session:
            return await send(session, dead, Request(0, [1], 1), 0.0, time.perf_counter())

    rec = asyncio.run(go())
    assert rec["status"] == "error" and rec["error"]


def test_the_mock_is_launched_with_the_harness_interpreter(mock_adapter):
    assert mock_adapter.launch_argv[0] == sys.executable


def test_an_event_with_empty_text_still_records_its_token_id_and_logprob():
    """A held-back multi-byte character streams an empty-text event; dropping it would make the
    reference score a sequence the engine never produced."""
    from bench.client.stream import _on_event

    rec = {"token_logprobs": [], "token_ids": []}
    times: list[float] = []
    empty = (b'{"choices":[{"text":"","logprobs":{"tokens":["token_id:7"],'
             b'"token_logprobs":[-1.5]}}]}')
    assert not _on_event(empty, rec, times, 0.5, parse_tokens=True)
    text = (b'{"choices":[{"text":" a","logprobs":{"tokens":["token_id:8"],'
            b'"token_logprobs":[-0.5]}}]}')
    _on_event(text, rec, times, 0.6, parse_tokens=True)
    assert rec["token_ids"] == [7, 8] and rec["token_logprobs"] == [-1.5, -0.5]
    assert times == [0.6]  # only the event with text is a streamed token for timing


def test_logprob_tokens_that_are_plain_text_give_no_ids_and_do_not_crash():
    from bench.client.stream import _on_event

    rec = {"token_logprobs": [], "token_ids": []}
    _on_event(b'{"choices":[{"text":" a","logprobs":{"tokens":[" a"],"token_logprobs":[-1.0]}}]}',
              rec, [], 0.1, parse_tokens=True)
    assert rec["token_ids"] == [] and rec["token_logprobs"] == [-1.0]
