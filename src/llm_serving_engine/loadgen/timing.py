"""Open-loop, coordinated-omission-aware load generator timing loop.

`intended_send_time` comes from a fixed arrival schedule and is recorded before `send_fn`
is awaited. A stall anywhere, sender side included, then shows up as a cluster of high
latencies; timing from the actual send, or waiting for one request before scheduling the
next, would hide it as fewer completed requests.

Arrival schedule: `next_send` starts at `time.monotonic()` and advances by
`random.expovariate(target_qps)` each iteration (Poisson arrivals). The loop sleeps
until `next_send` when early, records `intended_send_time = next_send`, then fires the
send as a fire-and-forget task (`asyncio.create_task`) so a slow request doesn't delay
the next scheduled arrival. Latency is `time.monotonic() - intended_send_time`,
computed once `send_fn` resolves and merged into the dict it returns (which must not
already contain a "latency" key).
"""

from __future__ import annotations

import asyncio
import random
import time
from typing import Awaitable, Callable


async def open_loop_load_gen(
    target_qps: float, duration_s: float, send_fn: Callable[[], Awaitable[dict]]
) -> list[dict]:
    next_send = time.monotonic()
    end = next_send + duration_s
    results: list[dict] = []
    tasks: list[asyncio.Task] = []
    while next_send < end:
        now = time.monotonic()
        if now < next_send:
            await asyncio.sleep(next_send - now)
        intended_send_time = next_send
        tasks.append(asyncio.create_task(_send_and_time(send_fn, intended_send_time, results)))
        next_send += random.expovariate(target_qps)
    await asyncio.gather(*tasks)  # requests still in flight at `end` must still be counted
    return results


async def _send_and_time(
    send_fn: Callable[[], Awaitable[dict]], intended_send_time: float, results: list[dict]
) -> None:
    result = await send_fn()
    results.append({"latency": time.monotonic() - intended_send_time, **result})
