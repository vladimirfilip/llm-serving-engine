"""open_loop_load_gen correctness: fixed arrival schedule independent of response time,
latency measured from the intended send time, and every started request's result
returned even if it's still in flight when the schedule ends.
"""

import asyncio
import time
from itertools import pairwise

import pytest

from llm_serving_engine.loadgen.timing import open_loop_load_gen


@pytest.mark.asyncio
async def test_fires_on_a_fixed_schedule(monkeypatch):
    monkeypatch.setattr("llm_serving_engine.loadgen.timing.random.expovariate", lambda qps: 0.05)

    async def send_fn() -> dict:
        return {}

    # next_send lands at offsets 0, 0.05, 0.10, 0.15, 0.20 within a 0.22s window.
    results = await open_loop_load_gen(target_qps=20.0, duration_s=0.22, send_fn=send_fn)
    assert len(results) == 5


@pytest.mark.asyncio
async def test_a_stalled_request_does_not_delay_later_arrivals(monkeypatch):
    monkeypatch.setattr("llm_serving_engine.loadgen.timing.random.expovariate", lambda qps: 0.05)
    call_times: list[float] = []

    async def send_fn() -> dict:
        call_times.append(time.monotonic())
        if len(call_times) == 1:
            await asyncio.sleep(0.15)  # outlives every remaining scheduled arrival
        return {}

    results = await open_loop_load_gen(target_qps=20.0, duration_s=0.22, send_fn=send_fn)

    assert len(results) == 5  # the stall didn't shrink the completed count
    gaps = [b - a for a, b in pairwise(call_times)]
    assert all(gap < 0.1 for gap in gaps)  # later sends kept their schedule through the stall


@pytest.mark.asyncio
async def test_latency_is_measured_from_intended_send_time_not_actual_completion():
    async def send_fn() -> dict:
        await asyncio.sleep(0.05)
        return {}

    results = await open_loop_load_gen(target_qps=1000.0, duration_s=0.01, send_fn=send_fn)
    assert results
    assert all(r["latency"] >= 0.05 for r in results)


@pytest.mark.asyncio
async def test_results_merge_send_fn_dict_under_the_latency_key():
    async def send_fn() -> dict:
        return {"success": True, "num_tokens_received": 3}

    results = await open_loop_load_gen(target_qps=1000.0, duration_s=0.01, send_fn=send_fn)
    assert results
    for r in results:
        assert r["success"] is True
        assert r["num_tokens_received"] == 3
        assert isinstance(r["latency"], float)


@pytest.mark.asyncio
async def test_a_request_still_in_flight_when_the_schedule_ends_is_still_returned(monkeypatch):
    monkeypatch.setattr("llm_serving_engine.loadgen.timing.random.expovariate", lambda qps: 0.05)

    async def send_fn() -> dict:
        await asyncio.sleep(0.1)  # outlives the whole arrival schedule below
        return {}

    results = await open_loop_load_gen(target_qps=20.0, duration_s=0.05, send_fn=send_fn)
    assert len(results) == 1
