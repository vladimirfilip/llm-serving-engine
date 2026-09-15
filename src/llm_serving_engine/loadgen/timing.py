"""Load generator timing loops.

Open loop, for latency: arrivals follow a fixed Poisson schedule, and each request's
`intended_send_time` is recorded before `send_fn` is awaited. A stall anywhere, sender side
included, shows up as a cluster of high latencies, and a server that can't keep up shows a
growing backlog. Latency is `time.monotonic() - intended_send_time`.

Closed loop, for maximum throughput: a fixed number of clients each send their next request
as soon as the previous one returns, so the server always has exactly that many requests
in flight. Its latencies measure a queue the loop itself bounds; only throughput counts.

Each result is `send_fn`'s dict plus "latency", "completed_at" and, open loop only,
"scheduled_at"; both offsets are seconds since the loop started.
"""

from __future__ import annotations

import asyncio
import random
import time
from typing import Awaitable, Callable

SendFn = Callable[[], Awaitable[dict]]


async def open_loop_load_gen(target_qps: float, duration_s: float, send_fn: SendFn) -> list[dict]:
    start = time.monotonic()
    next_send = start
    results: list[dict] = []
    tasks: list[asyncio.Task] = []
    while next_send < start + duration_s:
        now = time.monotonic()
        if now < next_send:
            await asyncio.sleep(next_send - now)
        tasks.append(asyncio.create_task(_send_on_schedule(send_fn, start, next_send, results)))
        next_send += random.expovariate(target_qps)
    await asyncio.gather(*tasks)  # requests still in flight at the end still count
    return results


async def closed_loop_load_gen(concurrency: int, duration_s: float, send_fn: SendFn) -> list[dict]:
    start = time.monotonic()
    results: list[dict] = []

    async def client() -> None:
        while time.monotonic() < start + duration_s:
            sent = time.monotonic()
            result = await send_fn()
            now = time.monotonic()
            results.append({"latency": now - sent, "completed_at": now - start, **result})

    await asyncio.gather(*(client() for _ in range(concurrency)))
    return results


async def _send_on_schedule(
    send_fn: SendFn, start: float, intended_send_time: float, results: list[dict]
) -> None:
    result = await send_fn()
    now = time.monotonic()
    results.append({
        "latency": now - intended_send_time,
        "scheduled_at": intended_send_time - start,
        "completed_at": now - start,
        **result,
    })
