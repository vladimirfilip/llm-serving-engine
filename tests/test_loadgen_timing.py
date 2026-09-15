"""Timing loops. Open loop: a fixed arrival schedule independent of response time, latency
from the intended send time, and every started request returned even if still in flight
when the schedule ends. Closed loop: a constant number of requests in flight."""

import asyncio
import random
import time
from itertools import pairwise

import pytest

from llm_serving_engine.loadgen.timing import closed_loop_load_gen, open_loop_load_gen


class FixedGaps:
    """An rng whose every exponential draw is `gap`."""

    def __init__(self, gap: float):
        self.gap = gap

    def expovariate(self, rate: float) -> float:
        return self.gap


@pytest.mark.asyncio
async def test_fires_on_a_fixed_schedule():

    async def send_fn() -> dict:
        return {}

    # next_send lands at offsets 0, 0.05, 0.10, 0.15, 0.20 within a 0.22s window.
    results = await open_loop_load_gen(20.0, 0.22, send_fn, FixedGaps(0.05))
    assert len(results) == 5


@pytest.mark.asyncio
async def test_a_stalled_request_does_not_delay_later_arrivals():
    call_times: list[float] = []

    async def send_fn() -> dict:
        call_times.append(time.monotonic())
        if len(call_times) == 1:
            await asyncio.sleep(0.15)  # outlives every remaining scheduled arrival
        return {}

    results = await open_loop_load_gen(20.0, 0.22, send_fn, FixedGaps(0.05))

    assert len(results) == 5  # the stall didn't shrink the completed count
    gaps = [b - a for a, b in pairwise(call_times)]
    assert all(gap < 0.1 for gap in gaps)  # later sends kept their schedule through the stall


@pytest.mark.asyncio
async def test_latency_is_measured_from_intended_send_time_not_actual_completion():
    async def send_fn() -> dict:
        await asyncio.sleep(0.05)
        return {}

    results = await open_loop_load_gen(1000.0, 0.01, send_fn, random.Random(0))
    assert results
    assert all(r["latency"] >= 0.05 for r in results)


@pytest.mark.asyncio
async def test_results_merge_send_fn_dict_under_the_latency_key():
    async def send_fn() -> dict:
        return {"success": True, "num_tokens_received": 3}

    results = await open_loop_load_gen(1000.0, 0.01, send_fn, random.Random(0))
    assert results
    for r in results:
        assert r["success"] is True
        assert r["num_tokens_received"] == 3
        assert isinstance(r["latency"], float)


@pytest.mark.asyncio
async def test_a_request_still_in_flight_when_the_schedule_ends_is_still_returned():

    async def send_fn() -> dict:
        await asyncio.sleep(0.1)  # outlives the whole arrival schedule below
        return {}

    results = await open_loop_load_gen(20.0, 0.05, send_fn, FixedGaps(0.05))
    assert len(results) == 1


@pytest.mark.asyncio
async def test_open_loop_results_record_schedule_and_completion_offsets():

    async def send_fn() -> dict:
        await asyncio.sleep(0.02)
        return {}

    results = await open_loop_load_gen(20.0, 0.12, send_fn, FixedGaps(0.05))

    scheduled = sorted(r["scheduled_at"] for r in results)
    assert scheduled == pytest.approx([0.0, 0.05, 0.10], abs=0.01)
    assert all(r["completed_at"] >= r["scheduled_at"] + 0.02 for r in results)


@pytest.mark.asyncio
async def test_closed_loop_keeps_exactly_concurrency_requests_in_flight():
    in_flight = 0
    peak = 0

    async def send_fn() -> dict:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return {}

    results = await closed_loop_load_gen(concurrency=3, duration_s=0.1, send_fn=send_fn)

    assert peak == 3
    assert len(results) >= 3 * 8  # each client sends back to back for the whole run
    assert all("scheduled_at" not in r and r["completed_at"] > 0 for r in results)


@pytest.mark.asyncio
async def test_first_token_latency_counts_from_the_intended_send_time():
    calls = 0

    async def send_fn() -> dict:
        nonlocal calls
        calls += 1
        if calls == 1:
            time.sleep(0.12)  # blocks the loop: the next arrivals go out late
        return {"token_times": [time.monotonic()]}

    results = await open_loop_load_gen(20.0, 0.12, send_fn, FixedGaps(0.05))

    late = [r for r in results if r["scheduled_at"] > 0]
    assert late
    assert all(
        r["first_token_latency"] >= 0.12 - r["scheduled_at"] - 0.005 for r in late
    )  # the stall counts, though each send got its token instantly


@pytest.mark.asyncio
async def test_a_request_with_no_tokens_has_no_first_token_latency():
    async def send_fn() -> dict:
        return {"token_times": []}

    results = await closed_loop_load_gen(concurrency=1, duration_s=0.01, send_fn=send_fn)
    assert all(r["first_token_latency"] is None for r in results)


@pytest.mark.asyncio
async def test_the_same_seed_offers_the_same_arrival_schedule():
    async def send_fn() -> dict:
        return {}

    async def schedule(seed: int) -> list[float]:
        results = await open_loop_load_gen(200.0, 0.1, send_fn, random.Random(seed))
        return sorted(r["scheduled_at"] for r in results)

    assert await schedule(7) == await schedule(7)
    assert await schedule(7) != await schedule(8)
