"""Scheduler thread -> IO thread token delivery.

Each iteration crosses to the event loop as one `call_soon_threadsafe` callback, so a batch
of any size wakes the loop once. `QueueFull` is caught inside that callback, on the loop.
"""

from __future__ import annotations

import asyncio
import itertools
from typing import Final

DONE: Final = object()
ABORTED: Final = object()

# seq_id -> per-request output queue. seq_ids come from one process-wide counter, so an
# id is never reused, even by an engine that replaced another.
output_channels: dict[int, asyncio.Queue] = {}
_seq_ids = itertools.count()


def new_output_channel(maxsize: int) -> tuple[int, asyncio.Queue]:
    """Bounded per-request channel: a slow client drops its own tokens, so delivery to
    every other channel never waits on it."""
    seq_id = next(_seq_ids)
    q: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
    output_channels[seq_id] = q
    return seq_id, q


def dispatch_results(
    iter_results: list[tuple[int, int, bool]], loop: asyncio.AbstractEventLoop
) -> None:
    """(seq_id, token, finished) from one iteration, handed over from a non-loop thread.
    The scheduler never touches `iter_results` again, so the loop may read it later."""
    if iter_results:
        loop.call_soon_threadsafe(_deliver_results, iter_results)


def dispatch_aborted(seq_ids: list[int], loop: asyncio.AbstractEventLoop) -> None:
    if seq_ids:
        loop.call_soon_threadsafe(_deliver_aborted, seq_ids)


def _deliver_results(iter_results: list[tuple[int, int, bool]]) -> None:
    for seq_id, token, finished in iter_results:
        q = output_channels.get(seq_id)
        if q is None:
            continue  # client disconnected
        try:
            q.put_nowait(token)
        except asyncio.QueueFull:
            pass
        if finished:
            _put_terminal(q, DONE)


def _deliver_aborted(seq_ids: list[int]) -> None:
    for seq_id in seq_ids:
        q = output_channels.get(seq_id)
        if q is not None:
            _put_terminal(q, ABORTED)


def _put_terminal(q: asyncio.Queue, sentinel: object) -> None:
    """The stream only ends once its sentinel lands, so a full queue gives up its oldest
    token to make room."""
    if q.full():
        q.get_nowait()
    q.put_nowait(sentinel)
