"""Scheduler correctness, run against every Scheduler implementation: decode-first
priority that's never budget-stalled, chunked-prefill progress and completion,
admission that backs off without skipping the queue when the KV pool is short, and
result handling that frees blocks and drops only finished sequences from `running`.
Admission-policy differences between implementations (continuous interleaving vs.
static batch-then-drain) are covered by dedicated per-class tests below.
"""

from collections import deque

import pytest

from llm_serving_engine.model.sampling import SamplingParams
from llm_serving_engine.observability.metrics import RequestMetrics
from llm_serving_engine.scheduling.allocator import BlockAllocator
from llm_serving_engine.scheduling.scheduler import (
    TOKEN_BUDGET,
    ContinuousBatchedScheduler,
    StaticBatchedScheduler,
)
from llm_serving_engine.scheduling.sequence import Sequence

SCHEDULER_CLASSES = [ContinuousBatchedScheduler, StaticBatchedScheduler]


def make_sequence(**overrides) -> Sequence:
    defaults = dict(
        seq_id=1,
        prompt_tokens=[1, 2, 3, 4, 5],
        sampling_params=SamplingParams(),
        arrival_time=0.0,
        metrics=RequestMetrics(enqueue_time=0.0),
    )
    defaults.update(overrides)
    return Sequence(**defaults)


def test_token_budget_default():
    assert TOKEN_BUDGET == 4096


@pytest.mark.parametrize("scheduler_cls", SCHEDULER_CLASSES)
def test_decoding_sequences_each_get_one_token_and_are_never_budget_stalled(scheduler_cls):
    scheduler = scheduler_cls(token_budget=2)  # smaller than the batch
    alloc = BlockAllocator(num_blocks=100, block_size=16)
    running = [
        make_sequence(seq_id=i, status="DECODING", generated_tokens=[9]) for i in range(5)
    ]

    plan = scheduler.scheduler_step(running=running, waiting=deque(), allocator=alloc)

    assert len(plan) == 5  # a stalled decode is the exact failure mode this ordering avoids
    assert {e.seq_id for e in plan} == {0, 1, 2, 3, 4}
    assert all(e.n_tokens == 1 and not e.is_prefill_chunk for e in plan)


@pytest.mark.parametrize("scheduler_cls", SCHEDULER_CLASSES)
def test_prefill_continues_from_progress_and_flips_to_decoding_on_completion(scheduler_cls):
    scheduler = scheduler_cls()
    alloc = BlockAllocator(num_blocks=100, block_size=16)
    seq = make_sequence(status="PREFILLING", prefill_progress=3)  # 2 of 5 tokens remain
    running = [seq]

    plan = scheduler.scheduler_step(running=running, waiting=deque(), allocator=alloc)

    [entry] = list(plan)
    assert entry.n_tokens == 2
    assert seq.prefill_progress == 5
    assert seq.status == "DECODING"


@pytest.mark.parametrize("scheduler_cls", SCHEDULER_CLASSES)
def test_prefill_chunk_is_capped_by_budget_and_stays_prefilling(scheduler_cls):
    scheduler = scheduler_cls(token_budget=2)
    alloc = BlockAllocator(num_blocks=100, block_size=16)
    seq = make_sequence(status="PREFILLING", prefill_progress=0)  # 5 prompt tokens
    running = [seq]

    plan = scheduler.scheduler_step(running=running, waiting=deque(), allocator=alloc)

    [entry] = list(plan)
    assert entry.n_tokens == 2
    assert seq.prefill_progress == 2
    assert seq.status == "PREFILLING"


@pytest.mark.parametrize("scheduler_cls", SCHEDULER_CLASSES)
def test_admission_moves_a_waiting_sequence_into_running(scheduler_cls):
    scheduler = scheduler_cls()
    alloc = BlockAllocator(num_blocks=100, block_size=16)
    seq = make_sequence()
    waiting = deque([seq])
    running = []

    plan = scheduler.scheduler_step(running=running, waiting=waiting, allocator=alloc)

    assert seq in running
    assert seq not in waiting
    [entry] = list(plan)
    assert entry.n_tokens == len(seq.prompt_tokens)
    assert seq.status == "DECODING"


@pytest.mark.parametrize("scheduler_cls", SCHEDULER_CLASSES)
def test_admission_backs_off_without_skipping_the_queue_when_pool_is_short(scheduler_cls):
    scheduler = scheduler_cls()
    alloc = BlockAllocator(num_blocks=0, block_size=16)  # nothing ever fits
    seq_a = make_sequence(seq_id=1)
    seq_b = make_sequence(seq_id=2)
    waiting = deque([seq_a, seq_b])
    running = []

    plan = scheduler.scheduler_step(running=running, waiting=waiting, allocator=alloc)

    assert len(plan) == 0
    assert list(waiting) == [seq_a, seq_b]  # head of the line, not skipped or dropped
    assert running == []


@pytest.mark.parametrize("scheduler_cls", SCHEDULER_CLASSES)
def test_handle_iteration_results_frees_blocks_and_drops_only_finished_sequences(
    scheduler_cls, monkeypatch
):
    monkeypatch.setattr("llm_serving_engine.scheduling.scheduler.dispatch_results", lambda *a, **k: None)
    scheduler = scheduler_cls()
    alloc = BlockAllocator(num_blocks=4, block_size=16)
    seq_a = make_sequence(seq_id=1, status="DECODING")
    seq_b = make_sequence(seq_id=2, status="DECODING")
    alloc.get_capacity(seq_a, new_tokens=1)
    alloc.get_capacity(seq_b, new_tokens=1)
    running = [seq_a, seq_b]

    scheduler.handle_iteration_results(
        [(1, 99, True), (2, 100, False)], running=running, allocator=alloc, loop=None
    )

    assert seq_a.generated_tokens == [99]
    assert seq_b.generated_tokens == [100]
    assert seq_a not in running
    assert seq_b in running
    assert seq_a.block_table.physical_blocks == []
    assert seq_b.block_table.physical_blocks != []


def test_continuous_scheduler_admits_new_work_while_others_are_already_running():
    scheduler = ContinuousBatchedScheduler()
    alloc = BlockAllocator(num_blocks=100, block_size=16)
    decoding = make_sequence(seq_id=1, status="DECODING", generated_tokens=[9])
    waiting_seq = make_sequence(seq_id=2)
    running = [decoding]
    waiting = deque([waiting_seq])

    plan = scheduler.scheduler_step(running=running, waiting=waiting, allocator=alloc)

    assert waiting_seq in running  # admitted alongside the already-running decode
    assert {e.seq_id for e in plan} == {1, 2}


def test_continuous_scheduler_admission_stops_at_the_concurrency_cap():
    scheduler = ContinuousBatchedScheduler(max_concurrent_sequences=1)
    alloc = BlockAllocator(num_blocks=100, block_size=16)
    decoding = make_sequence(seq_id=1, status="DECODING", generated_tokens=[9])
    waiting_seq = make_sequence(seq_id=2)
    running = [decoding]
    waiting = deque([waiting_seq])

    plan = scheduler.scheduler_step(running=running, waiting=waiting, allocator=alloc)

    assert waiting_seq not in running  # cap already met by the running decode
    assert list(waiting) == [waiting_seq]
    assert {e.seq_id for e in plan} == {1}


def test_static_scheduler_blocks_admission_until_the_running_batch_fully_drains():
    scheduler = StaticBatchedScheduler()
    alloc = BlockAllocator(num_blocks=100, block_size=16)
    decoding = make_sequence(seq_id=1, status="DECODING", generated_tokens=[9])
    waiting_seq = make_sequence(seq_id=2)
    running = [decoding]
    waiting = deque([waiting_seq])

    plan = scheduler.scheduler_step(running=running, waiting=waiting, allocator=alloc)

    assert waiting_seq not in running  # batch is still in flight, no new admissions
    assert list(waiting) == [waiting_seq]
    assert {e.seq_id for e in plan} == {1}


def test_static_scheduler_admits_next_batch_once_running_is_empty():
    scheduler = StaticBatchedScheduler(batch_size=2)
    alloc = BlockAllocator(num_blocks=100, block_size=16)
    seq_a = make_sequence(seq_id=1)
    seq_b = make_sequence(seq_id=2)
    seq_c = make_sequence(seq_id=3)
    running = []
    waiting = deque([seq_a, seq_b, seq_c])

    plan = scheduler.scheduler_step(running=running, waiting=waiting, allocator=alloc)

    assert running == [seq_a, seq_b]  # capped at batch_size, third stays queued
    assert list(waiting) == [seq_c]
    assert {e.seq_id for e in plan} == {1, 2}
