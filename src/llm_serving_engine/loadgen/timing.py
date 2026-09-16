"""Load generator timing loops.

Both loops send a fixed `num_requests`, so a run's sample size is what it was asked for
rather than whatever the server's speed allowed; `max_duration_s` cuts a run short when an
arm is too slow to finish its count in reasonable time.

Open loop, for latency: arrivals follow a fixed Poisson schedule, and each request's
`intended_send_time` is recorded before `send_fn` is awaited. A stall anywhere, sender side
included, shows up as a cluster of high latencies, and a server that can't keep up shows a
growing backlog. Latency and time to first token both count from `intended_send_time`.
The schedule is drawn from `rng`, so a seeded run offers the same arrivals to every config.

Closed loop, for maximum throughput: a fixed number of clients divide `num_requests` between
them, each sending its next as soon as the previous one returns, so the server has that many
requests in flight until the count runs out. Its latencies measure a queue the loop itself
bounds; only throughput counts.

Each result is `send_fn`'s dict plus "latency", "first_token_latency" (None if no token
arrived), "completed_at" and, open loop only, "scheduled_at"; both offsets are seconds since
the loop started.
"""

from __future__ import annotations

import asyncio
import random
import time
from typing import Awaitable, Callable

SendFn = Callable[[], Awaitable[dict]]


async def open_loop_load_gen(
    target_qps: float,
    num_requests: int,
    send_fn: SendFn,
    rng: random.Random,
    max_duration_s: float | None = None,
) -> list[dict]:
    start = time.monotonic()
    next_send = start
    results: list[dict] = []
    tasks: list[asyncio.Task] = []
    for _ in range(num_requests):
        if max_duration_s is not None and next_send - start >= max_duration_s:
            break
        now = time.monotonic()
        if now < next_send:
            await asyncio.sleep(next_send - now)
        tasks.append(asyncio.create_task(_send_on_schedule(send_fn, start, next_send, results)))
        next_send += rng.expovariate(target_qps)
    await asyncio.gather(*tasks)  # requests still in flight at the end still count
    return results


async def closed_loop_load_gen(
    concurrency: int, num_requests: int, send_fn: SendFn, max_duration_s: float | None = None
) -> list[dict]:
    start = time.monotonic()
    results: list[dict] = []
    unsent = num_requests

    async def client() -> None:
        nonlocal unsent
        while unsent > 0:
            if max_duration_s is not None and time.monotonic() - start >= max_duration_s:
                return
            # Claimed before the await, so the clients divide `num_requests` between them.
            unsent -= 1
            sent = time.monotonic()
            result = await send_fn()
            results.append(_timed(result, start, sent))

    await asyncio.gather(*(client() for _ in range(concurrency)))
    return results


async def _send_on_schedule(
    send_fn: SendFn, start: float, intended_send_time: float, results: list[dict]
) -> None:
    result = await send_fn()
    timed = _timed(result, start, intended_send_time)
    results.append({"scheduled_at": intended_send_time - start, **timed})


def _timed(result: dict, start: float, sent: float) -> dict:
    """`result` with its latencies measured from `sent`, and its completion offset."""
    now = time.monotonic()
    token_times = result.get("token_times") or []
    return {
        "latency": now - sent,
        "first_token_latency": token_times[0] - sent if token_times else None,
        "completed_at": now - start,
        **result,
    }
