import asyncio
import threading

import pytest

from llm_serving_engine.scheduling import dispatch
from llm_serving_engine.scheduling.dispatch import DONE, dispatch_results, new_output_channel


@pytest.fixture(autouse=True)
def _clear_output_channels():
    dispatch.output_channels.clear()
    yield
    dispatch.output_channels.clear()


@pytest.mark.asyncio
async def test_dispatch_results_delivers_token_and_done_on_finish():
    q = new_output_channel(seq_id=1, maxsize=64)
    loop = asyncio.get_running_loop()

    dispatch_results([(1, 42, True)], loop)
    await asyncio.sleep(0)  # let call_soon_threadsafe callbacks run

    assert await q.get() == 42
    assert await q.get() is DONE


@pytest.mark.asyncio
async def test_dispatch_from_a_non_loop_thread_delivers():
    """The real caller is the GPU worker, a plain OS thread — the case call_soon_threadsafe
    exists for, and the one an on-loop call can't distinguish from a raw put_nowait."""
    q = new_output_channel(seq_id=7, maxsize=64)
    loop = asyncio.get_running_loop()

    worker = threading.Thread(target=dispatch_results, args=([(7, 5, True)], loop))
    worker.start()
    worker.join()

    assert await asyncio.wait_for(q.get(), timeout=1) == 5
    assert await asyncio.wait_for(q.get(), timeout=1) is DONE


@pytest.mark.asyncio
async def test_dispatch_results_skips_unknown_seq_id():
    loop = asyncio.get_running_loop()
    dispatch_results([(999, 1, False)], loop)
    await asyncio.sleep(0)
    assert 999 not in dispatch.output_channels


@pytest.mark.asyncio
async def test_full_queue_drops_rather_than_raises():
    q = new_output_channel(seq_id=2, maxsize=4)
    for i in range(4):
        q.put_nowait(i)

    loop = asyncio.get_running_loop()
    dispatch_results([(2, 999, False)], loop)
    await asyncio.sleep(0)  # the dropped put must not raise into the event loop

    assert q.full()
    assert q.get_nowait() == 0  # original contents untouched, new token was dropped


def test_new_output_channel_registers_bounded_queue_under_seq_id():
    q = new_output_channel(seq_id=3, maxsize=8)
    assert dispatch.output_channels[3] is q
    assert q.maxsize == 8
