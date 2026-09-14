import asyncio
import threading

import pytest

from llm_serving_engine.scheduling.dispatch import (
    ABORTED,
    DONE,
    dispatch_aborted,
    dispatch_results,
    new_output_channel,
    output_channels,
)


@pytest.mark.asyncio
async def test_finished_result_delivers_token_then_done():
    seq_id, q = new_output_channel(maxsize=64)

    dispatch_results([(seq_id, 42, True)], asyncio.get_running_loop())
    await asyncio.sleep(0)

    assert q.get_nowait() == 42
    assert q.get_nowait() is DONE


@pytest.mark.asyncio
async def test_delivery_from_a_non_loop_thread():
    seq_id, q = new_output_channel(maxsize=64)
    loop = asyncio.get_running_loop()

    worker = threading.Thread(target=dispatch_results, args=([(seq_id, 5, True)], loop))
    worker.start()
    worker.join()

    assert await asyncio.wait_for(q.get(), timeout=1) == 5
    assert await asyncio.wait_for(q.get(), timeout=1) is DONE


@pytest.mark.asyncio
async def test_result_for_a_disconnected_client_is_dropped():
    dispatch_results([(999, 1, False)], asyncio.get_running_loop())
    await asyncio.sleep(0)
    assert 999 not in output_channels


@pytest.mark.asyncio
async def test_full_queue_drops_the_new_token():
    seq_id, q = new_output_channel(maxsize=4)
    for i in range(4):
        q.put_nowait(i)

    dispatch_results([(seq_id, 999, False)], asyncio.get_running_loop())
    await asyncio.sleep(0)

    assert [q.get_nowait() for _ in range(4)] == [0, 1, 2, 3]


@pytest.mark.asyncio
async def test_done_lands_on_a_full_queue():
    seq_id, q = new_output_channel(maxsize=2)
    q.put_nowait(1)
    q.put_nowait(2)

    dispatch_results([(seq_id, 3, True)], asyncio.get_running_loop())
    await asyncio.sleep(0)

    assert q.qsize() == 2
    assert [q.get_nowait() for _ in range(2)][-1] is DONE


@pytest.mark.asyncio
async def test_aborted_lands_on_a_full_queue():
    seq_id, q = new_output_channel(maxsize=1)
    q.put_nowait(1)

    dispatch_aborted([seq_id], asyncio.get_running_loop())
    await asyncio.sleep(0)

    assert q.get_nowait() is ABORTED


def test_new_output_channel_registers_a_bounded_queue_under_a_fresh_seq_id():
    first, q = new_output_channel(maxsize=8)
    second, _ = new_output_channel(maxsize=8)
    assert output_channels[first] is q
    assert q.maxsize == 8
    assert second != first
