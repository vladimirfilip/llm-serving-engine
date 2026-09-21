"""Open-loop and closed-loop runners. Open loop sends each request at its scheduled time
whether or not earlier ones finished; closed loop keeps a fixed number in flight. Requests
shard across client processes by index, all sharing one `t0`."""

from __future__ import annotations

import asyncio
import contextlib
import multiprocessing
import os
import time
from typing import Callable, Iterator

import pandas as pd

from ..workloads.workloads import Request
from .stream import Target, new_session, send

SHARD_START_SLACK_S = 0.5
LOOP_START_SLACK_S = 0.05


def _use_uvloop() -> None:
    try:
        import uvloop

        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    except ImportError:
        pass


def uvloop_available() -> bool:
    try:
        import uvloop  # noqa: F401
    except ImportError:
        return False
    return True


async def open_loop(target: Target, requests: list[Request], t_sched: list[float],
                    t0: float) -> list[dict]:
    async with new_session() as session:
        return list(await asyncio.gather(
            *(send(session, target, r, t, t0) for r, t in zip(requests, t_sched, strict=True))))


def _warm(_index: int) -> None:
    """Holds a worker long enough that each worker of the pool takes exactly one, so every
    process has finished importing before the shared clock starts."""
    time.sleep(0.3)


def _shard_main(args: tuple) -> list[dict]:
    target, requests, t_sched, t0, cpus = args
    if cpus:
        os.sched_setaffinity(0, cpus)
    _use_uvloop()
    return asyncio.run(open_loop(target, requests, t_sched, t0))


@contextlib.contextmanager
def pinned(cpus: set[int] | None) -> Iterator[None]:
    """Pins this process to `cpus` for the block, then restores its affinity."""
    before = os.sched_getaffinity(0)
    if cpus:
        os.sched_setaffinity(0, cpus)
    try:
        yield
    finally:
        os.sched_setaffinity(0, before)


def parse_cpus(spec: str | None) -> set[int] | None:
    """`"5-8"` or `"0,2-3"` to a set of CPU ids."""
    if not spec:
        return None
    cpus: set[int] = set()
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        cpus.update(range(int(lo), int(hi or lo) + 1))
    return cpus


def run_open_loop(target: Target, requests: list[Request], t_sched: list[float], procs: int = 1,
                  cpus: str | None = None, t0: float | None = None) -> tuple[list[dict], float]:
    """Returns the records sorted by `req_id` and the shared `t0`. Request `i` goes to process
    `i % procs`."""
    affinity = parse_cpus(cpus)
    if procs == 1:
        _use_uvloop()
        t0 = t0 if t0 is not None else time.perf_counter() + LOOP_START_SLACK_S
        with pinned(affinity):
            records = asyncio.run(open_loop(target, requests, t_sched, t0))
        return sorted(records, key=_req_id), t0
    with multiprocessing.get_context("spawn").Pool(procs) as pool:
        pool.map(_warm, range(procs), chunksize=1)
        t0 = t0 if t0 is not None else time.perf_counter() + SHARD_START_SLACK_S
        shards = [(target, requests[i::procs], t_sched[i::procs], t0, affinity)
                  for i in range(procs)]
        merged = [rec for shard in pool.map(_shard_main, shards, chunksize=1) for rec in shard]
    return sorted(merged, key=_req_id), t0


def _req_id(record: dict) -> int:
    return record["req_id"]


async def closed_loop(target: Target, next_request: Callable[[int], Request], concurrency: int,
                      duration_s: float, t0: float) -> list[dict]:
    """`concurrency` workers, each reissuing the moment its request finishes, for `duration_s`.
    Requests in flight when time runs out finish and are recorded."""
    records: list[dict] = []
    end = t0 + duration_s

    async def worker(index: int, session) -> None:
        while time.perf_counter() < end:
            request = next_request(index)
            records.append(await send(session, target, request, 0.0, t0, open_loop=False))

    async with new_session() as session:
        await asyncio.gather(*(worker(i, session) for i in range(concurrency)))
    return records


async def bounded(target: Target, requests: list[Request], concurrency: int,
                  t0: float) -> list[dict]:
    """Every request once, at most `concurrency` in flight; each starts as a slot frees."""
    slots = asyncio.Semaphore(concurrency)

    async def one(session, req: Request) -> dict:
        async with slots:
            return await send(session, target, req, 0.0, t0, open_loop=False)

    async with new_session() as session:
        return list(await asyncio.gather(*(one(session, r) for r in requests)))


def run_bounded(target: Target, requests: list[Request], concurrency: int,
                cpus: str | None = None) -> tuple[list[dict], float]:
    _use_uvloop()
    t0 = time.perf_counter()
    with pinned(parse_cpus(cpus)):
        records = asyncio.run(bounded(target, requests, concurrency, t0))
    return sorted(records, key=_req_id), t0


def run_closed_loop(target: Target, next_request: Callable[[int], Request], concurrency: int,
                    duration_s: float, cpus: str | None = None) -> tuple[list[dict], float]:
    _use_uvloop()
    t0 = time.perf_counter()
    with pinned(parse_cpus(cpus)):
        records = asyncio.run(closed_loop(target, next_request, concurrency, duration_s, t0))
    return records, t0


def records_frame(records: list[dict], **columns) -> pd.DataFrame:
    """One row per request, tagged with run identifiers."""
    df = pd.DataFrame(records)
    for name, value in columns.items():
        df[name] = value
    return df
