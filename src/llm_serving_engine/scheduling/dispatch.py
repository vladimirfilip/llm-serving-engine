"""GPU worker -> IO thread token delivery.

`call_soon_threadsafe` schedules a callback and returns immediately, so the
`except QueueFull` guard must live inside the callback that runs on the loop, not
around the call that schedules it.
"""

from __future__ import annotations

import asyncio
from typing import Final

DONE: Final = object()

# seq_id -> per-request output queue.
output_channels: dict[int, asyncio.Queue] = {}


def new_output_channel(seq_id: int, maxsize: int) -> asyncio.Queue:
    """Bounded per-request channel: a slow client drops tokens instead of stalling
    delivery to every other sequence's channel."""
    q: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
    output_channels[seq_id] = q
    return q


def _safe_put(q: asyncio.Queue, item: object) -> None:
    try:
        q.put_nowait(item)
    except asyncio.QueueFull:
        pass  # slow client; drop rather than block the GPU worker


def dispatch_results(
    iter_results: list[tuple[int, int, bool]],
    loop: asyncio.AbstractEventLoop,
) -> None:
    """Called from the GPU worker thread after each forward() with (seq_id, token, finished).

    Never call q.put_nowait directly from this thread: asyncio.Queue isn't thread-safe to
    push into from a thread that isn't running its event loop.
    """
    for seq_id, token, finished in iter_results:
        q = output_channels.get(seq_id)
        if q is None:
            continue  # client already disconnected
        loop.call_soon_threadsafe(_safe_put, q, token)
        if finished:
            loop.call_soon_threadsafe(_safe_put, q, DONE)
