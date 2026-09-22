"""Scheduler behaviour, run against every Scheduler implementation where the policies
agree: decode-first planning that TOKEN_BUDGET never stalls, chunked prefill, admission that
skips a head it can't fit until that head's skips run out, recompute preemption from the
tail of `running`, and result handling. Where the
policies differ (continuous interleaving vs. static batch-then-drain), per-class tests
cover each.
"""

from collections import deque

import pytest

from llm_serving_engine.scheduling.allocator import BlockAllocator
from llm_serving_engine.scheduling.scheduler import (
    MAX_ADMISSION_SKIPS,
    ContinuousBatchedScheduler,
    StaticBatchedScheduler,
)
from tests.factories import decoding_sequence, make_sequence, prefilling_sequence

SCHEDULER_CLASSES = [ContinuousBatchedScheduler, StaticBatchedScheduler]


class RecordingLoop:
    """Stands in for the event loop: records the delivery callbacks it is handed."""

    def __init__(self):
        self.callbacks = []

    def call_soon_threadsafe(self, callback, *args):
        self.callbacks.append((callback, args))


@pytest.mark.parametrize("scheduler_cls", SCHEDULER_CLASSES)
def test_every_decoding_sequence_gets_a_token_even_past_the_token_budget(scheduler_cls):
    scheduler = scheduler_cls(token_budget=2)
    alloc = BlockAllocator(num_blocks=100, block_size=16)
    running = [decoding_sequence(alloc, seq_id=i, prompt_len=5) for i in range(5)]

    plan = scheduler.scheduler_step(running=running, waiting=deque(), allocator=alloc)

    assert {e.seq_id for e in plan} == {0, 1, 2, 3, 4}
    assert all(e.n_tokens == 1 and not e.is_prefill_chunk for e in plan)


@pytest.mark.parametrize("scheduler_cls", SCHEDULER_CLASSES)
def test_prefill_continues_from_its_progress_and_flips_to_decoding_when_done(scheduler_cls):
    scheduler = scheduler_cls()
    alloc = BlockAllocator(num_blocks=100, block_size=16)
    seq = make_sequence(status="PREFILLING", prefill_progress=3)  # 2 of 5 tokens remain
    alloc.allocate(seq, 3)

    plan = scheduler.scheduler_step(running=[seq], waiting=deque(), allocator=alloc)

    [entry] = list(plan)
    assert entry.n_tokens == 2
    assert seq.prefill_progress == 5
    assert seq.status == "DECODING"


@pytest.mark.parametrize("scheduler_cls", SCHEDULER_CLASSES)
def test_one_prefill_chunk_takes_only_a_fraction_of_the_token_budget(scheduler_cls):
    scheduler = scheduler_cls(token_budget=8)  # chunks of at most 4
    alloc = BlockAllocator(num_blocks=100, block_size=16)
    seq = make_sequence(status="PREFILLING", prompt_tokens=[0] * 10)

    plan = scheduler.scheduler_step(running=[seq], waiting=deque(), allocator=alloc)

    [entry] = list(plan)
    assert entry.n_tokens == 4
    assert seq.prefill_progress == 4
    assert seq.status == "PREFILLING"


def test_a_long_prefill_leaves_budget_to_admit_a_waiting_prompt():
    scheduler = ContinuousBatchedScheduler(token_budget=8)  # chunks of at most 4
    alloc = BlockAllocator(num_blocks=100, block_size=16)
    prefilling = make_sequence(seq_id=1, status="PREFILLING", prompt_tokens=[0] * 40)
    newcomer = make_sequence(seq_id=2, prompt_tokens=[0] * 3)
    running, waiting = [prefilling], deque([newcomer])

    plan = scheduler.scheduler_step(running, waiting, alloc)

    assert [(e.seq_id, e.n_tokens) for e in plan] == [(1, 4), (2, 3)]
    assert running == [prefilling, newcomer]


@pytest.mark.parametrize("scheduler_cls", SCHEDULER_CLASSES)
def test_prefill_chunk_without_free_blocks_waits_without_losing_progress(scheduler_cls):
    scheduler = scheduler_cls()
    alloc = BlockAllocator(num_blocks=2, block_size=4)
    seq = make_sequence(prompt_tokens=[0] * 7, status="PREFILLING", prefill_progress=4)
    alloc.allocate(seq, 4)
    alloc.allocate(make_sequence(seq_id=99), 4)  # holds the other block

    plan = scheduler.scheduler_step(running=[seq], waiting=deque(), allocator=alloc)

    assert len(plan) == 0
    assert seq.prefill_progress == 4
    assert seq.block_table.num_tokens == 4


@pytest.mark.parametrize("scheduler_cls", SCHEDULER_CLASSES)
def test_prefills_that_together_exceed_the_pool_do_not_stall_each_other(scheduler_cls):
    alloc = BlockAllocator(num_blocks=4, block_size=4)
    oldest = prefilling_sequence(alloc, seq_id=1, prompt_len=12, progress=8)
    latest = prefilling_sequence(alloc, seq_id=2, prompt_len=12, progress=8)
    running, waiting = [oldest, latest], deque()

    plan = scheduler_cls().scheduler_step(running, waiting, alloc)

    assert [e.seq_id for e in plan] == [oldest.seq_id]
    assert oldest.status == "DECODING"
    assert plan.preempted == [latest.seq_id]
    assert running == [oldest]
    assert list(waiting) == [latest]


def test_a_preempted_prefill_frees_its_blocks_and_restarts_from_zero():
    alloc = BlockAllocator(num_blocks=4, block_size=4)
    oldest = prefilling_sequence(alloc, seq_id=1, prompt_len=12, progress=8)
    latest = prefilling_sequence(alloc, seq_id=2, prompt_len=12, progress=8)

    ContinuousBatchedScheduler().scheduler_step([oldest, latest], deque(), alloc)

    assert latest.status == "WAITING"
    assert latest.prefill_progress == 0
    assert latest.block_table.physical_blocks == []
    assert latest.block_table.num_tokens == 0


def test_a_preempted_prefills_progress_is_not_recovered_in_the_same_step():
    """A preemption zeroes the step's remaining budget even on success: without that, the
    victim could be re-admitted onto the very blocks it just freed, in the same step whose
    `plan.preempted` reports it lost its progress."""
    scheduler = ContinuousBatchedScheduler(token_budget=8)  # chunks of at most 4
    alloc = BlockAllocator(num_blocks=11, block_size=4)
    oldest = prefilling_sequence(alloc, seq_id=1, prompt_len=40, progress=8)
    latest = prefilling_sequence(alloc, seq_id=2, prompt_len=20, progress=16)
    alloc.allocate(make_sequence(seq_id=99), 20)  # holds the other 5 blocks
    running, waiting = [oldest, latest], deque()

    plan = scheduler.scheduler_step(running, waiting, alloc)

    assert [e.seq_id for e in plan] == [oldest.seq_id]
    assert plan.preempted == [latest.seq_id]
    assert list(waiting) == [latest]
    assert latest.prefill_progress == 0


def test_a_stalled_prefill_waits_for_decoders_instead_of_preempting_them():
    """Also the only test where the PREFILLING-only victim filter matters: without it, this
    would preempt the decoder, which already has a plan entry from step 1 -- the engine
    would then run that entry against a sequence dropped from `running`."""
    alloc = BlockAllocator(num_blocks=3, block_size=4)
    prefill = prefilling_sequence(alloc, seq_id=1, prompt_len=12, progress=8)
    decoder = decoding_sequence(alloc, seq_id=2, prompt_len=3)
    running, waiting = [prefill, decoder], deque()

    plan = ContinuousBatchedScheduler().scheduler_step(running, waiting, alloc)

    assert [e.seq_id for e in plan] == [decoder.seq_id]
    assert plan.preempted == []
    assert running == [prefill, decoder]


def test_a_stalled_prefill_preempts_the_latest_prefill_behind_it_first():
    alloc = BlockAllocator(num_blocks=4, block_size=4)
    first = prefilling_sequence(alloc, seq_id=1, prompt_len=12, progress=4)
    second = prefilling_sequence(alloc, seq_id=2, prompt_len=12, progress=4)
    third = prefilling_sequence(alloc, seq_id=3, prompt_len=12, progress=8)
    running, waiting = [first, second, third], deque()

    plan = ContinuousBatchedScheduler().scheduler_step(running, waiting, alloc)

    assert [e.seq_id for e in plan] == [first.seq_id]
    assert plan.preempted == [third.seq_id]
    assert running == [first, second]
    assert list(waiting) == [third]


def test_running_sequences_always_yield_a_plan_until_fully_drained():
    alloc = BlockAllocator(num_blocks=6, block_size=4)
    running = [prefilling_sequence(alloc, i, prompt_len=16, progress=8) for i in (1, 2, 3)]
    waiting = deque()
    scheduler = ContinuousBatchedScheduler()

    for _ in range(40):
        plan = scheduler.scheduler_step(running, waiting, alloc)
        assert running == [] or len(plan) > 0
        for seq in list(running):
            if seq.status == "DECODING":
                alloc.free(seq.block_table)
                running.remove(seq)
        if not running and not waiting:
            break
    assert not running and not waiting, "never drained: a live-lock hid behind a non-empty plan"


def test_a_blocked_prefill_continuation_stops_admission_behind_it():
    alloc = BlockAllocator(num_blocks=6, block_size=4)
    long_prompt = make_sequence(seq_id=1, prompt_tokens=[0] * 20, status="PREFILLING")
    alloc.allocate(long_prompt, 4)
    long_prompt.prefill_progress = 4
    alloc.allocate(make_sequence(seq_id=99), 12)  # 2 blocks left; the next chunk needs 4
    newcomer = make_sequence(seq_id=2, prompt_tokens=[0] * 2)  # 1 block + 1 of headroom
    running, waiting = [long_prompt], deque([newcomer])

    plan = ContinuousBatchedScheduler().scheduler_step(running, waiting, alloc)

    assert len(plan) == 0
    assert running == [long_prompt]
    assert list(waiting) == [newcomer]


@pytest.mark.parametrize("scheduler_cls", SCHEDULER_CLASSES)
def test_admission_moves_the_head_of_waiting_into_running(scheduler_cls):
    scheduler = scheduler_cls()
    alloc = BlockAllocator(num_blocks=100, block_size=16)
    seq = make_sequence()
    waiting, running = deque([seq]), []

    plan = scheduler.scheduler_step(running=running, waiting=waiting, allocator=alloc)

    assert running == [seq]
    assert not waiting
    [entry] = list(plan)
    assert entry.n_tokens == len(seq.prompt_tokens)
    assert seq.status == "DECODING"
    assert seq.metrics.admit_time is not None


@pytest.mark.parametrize("scheduler_cls", SCHEDULER_CLASSES)
def test_admission_skips_a_head_that_does_not_fit(scheduler_cls):
    scheduler = scheduler_cls()
    alloc = BlockAllocator(num_blocks=3, block_size=4)
    alloc.allocate(make_sequence(seq_id=99), 8)  # 1 block left
    head = make_sequence(seq_id=1, prompt_tokens=[0] * 5)  # needs 2 blocks now
    small = make_sequence(seq_id=2, prompt_tokens=[0] * 2)  # fits in 1
    waiting, running = deque([head, small]), []

    plan = scheduler.scheduler_step(running=running, waiting=waiting, allocator=alloc)

    assert [e.seq_id for e in plan] == [small.seq_id]
    assert running == [small]
    assert list(waiting) == [head]
    assert head.admission_skips == 1


def test_admission_freezes_once_the_head_has_spent_its_skips():
    scheduler = ContinuousBatchedScheduler()
    alloc = BlockAllocator(num_blocks=100, block_size=4)
    held = decoding_sequence(alloc, seq_id=99, prompt_len=40)
    head = make_sequence(seq_id=1, prompt_tokens=[0] * 380)  # fits the pool, not the free blocks
    running, waiting = [held], deque([head])

    admitted = []
    for seq_id in range(2, 2 + MAX_ADMISSION_SKIPS + 1):
        waiting.append(make_sequence(seq_id=seq_id, prompt_tokens=[0] * 2))
        plan = scheduler.scheduler_step(running, waiting, alloc)
        admitted += [e.seq_id for e in plan if e.is_prefill_chunk]

    # Every small arrival is admitted ahead of the head until its skips run out; the last
    # one waits behind it instead.
    assert admitted == list(range(2, 2 + MAX_ADMISSION_SKIPS))
    assert head.admission_skips == MAX_ADMISSION_SKIPS
    assert waiting[0] is head


@pytest.mark.parametrize("scheduler_cls", SCHEDULER_CLASSES)
def test_a_sequence_the_whole_pool_cannot_hold_is_rejected(scheduler_cls):
    scheduler = scheduler_cls()
    alloc = BlockAllocator(num_blocks=2, block_size=4)
    too_long = make_sequence(seq_id=1, prompt_tokens=[0] * 8)  # 9 slots, 3 blocks
    fits = make_sequence(seq_id=2, prompt_tokens=[0] * 3)
    waiting, running = deque([too_long, fits]), []

    plan = scheduler.scheduler_step(running=running, waiting=waiting, allocator=alloc)

    assert plan.rejected == [too_long]
    assert running == [fits]
    assert not waiting


def test_admission_keeps_a_free_block_for_each_running_sequence():
    alloc = BlockAllocator(num_blocks=3, block_size=4)
    running = [decoding_sequence(alloc, seq_id=1, prompt_len=3)]  # 1 block
    newcomer = make_sequence(seq_id=2, prompt_tokens=[0] * 8)  # 2 blocks
    waiting = deque([newcomer])

    ContinuousBatchedScheduler().scheduler_step(running=running, waiting=waiting, allocator=alloc)

    # 2 free blocks cover the newcomer, but not also the running sequence's next block.
    assert list(waiting) == [newcomer]


@pytest.mark.parametrize("scheduler_cls", SCHEDULER_CLASSES)
def test_decode_without_a_free_block_preempts_the_latest_arrival(scheduler_cls):
    scheduler = scheduler_cls()
    alloc = BlockAllocator(num_blocks=2, block_size=4)
    oldest = decoding_sequence(alloc, seq_id=1, prompt_len=4)  # block full
    latest = decoding_sequence(alloc, seq_id=2, prompt_len=4)  # block full, pool empty
    running, waiting = [oldest, latest], deque()

    plan = scheduler.scheduler_step(running=running, waiting=waiting, allocator=alloc)

    assert [e.seq_id for e in plan] == [oldest.seq_id]
    assert plan.preempted == [latest.seq_id]
    assert running == [oldest]
    assert list(waiting) == [latest]
    assert latest.status == "WAITING"
    assert latest.prefill_progress == 0
    assert latest.block_table.physical_blocks == []
    assert latest.block_table.num_tokens == 0


@pytest.mark.parametrize("scheduler_cls", SCHEDULER_CLASSES)
def test_preempted_sequences_return_to_the_head_of_waiting_in_arrival_order(scheduler_cls):
    scheduler = scheduler_cls()
    alloc = BlockAllocator(num_blocks=3, block_size=4)
    a, b, c = (decoding_sequence(alloc, seq_id=i, prompt_len=4) for i in (1, 2, 3))
    queued = make_sequence(seq_id=4)
    running, waiting = [a, b, c], deque([queued])

    plan = scheduler.scheduler_step(running=running, waiting=waiting, allocator=alloc)

    # a takes c's block; b then finds none and, now the tail, preempts itself.
    assert [e.seq_id for e in plan] == [a.seq_id]
    assert running == [a]
    assert list(waiting) == [b, c, queued]


def test_a_preempted_sequence_re_prefills_its_prompt_and_generated_tokens():
    alloc = BlockAllocator(num_blocks=10, block_size=4)
    seq = make_sequence(prompt_tokens=[1, 2, 3, 4], generated_tokens=[9, 10])
    running, waiting = [], deque([seq])

    plan = ContinuousBatchedScheduler().scheduler_step(running, waiting, alloc)

    [entry] = list(plan)
    assert entry.is_prefill_chunk
    assert entry.n_tokens == 6
    assert seq.status == "DECODING"
    assert seq.block_table.num_tokens == 6


def test_a_sequence_that_outgrows_the_whole_pool_is_preempted_then_rejected():
    scheduler = ContinuousBatchedScheduler()
    alloc = BlockAllocator(num_blocks=1, block_size=4)
    seq = decoding_sequence(alloc, seq_id=1, prompt_len=4)
    running, waiting = [seq], deque()

    plan = scheduler.scheduler_step(running, waiting, alloc)

    assert plan.preempted == [seq.seq_id]
    assert plan.rejected == [seq]
    assert not running
    assert not waiting


@pytest.mark.parametrize("scheduler_cls", SCHEDULER_CLASSES)
def test_handle_iteration_results_frees_and_returns_only_finished_sequences(scheduler_cls):
    scheduler = scheduler_cls()
    alloc = BlockAllocator(num_blocks=4, block_size=16)
    seq_a = decoding_sequence(alloc, seq_id=1, prompt_len=2)
    seq_b = decoding_sequence(alloc, seq_id=2, prompt_len=2)
    running = [seq_a, seq_b]

    finished = scheduler.handle_iteration_results(
        [(1, 99, True), (2, 100, False)], running=running, allocator=alloc, loop=RecordingLoop()
    )

    assert finished == [seq_a]
    assert running == [seq_b]
    assert seq_a.generated_tokens == [9, 99]
    assert seq_b.generated_tokens == [9, 100]
    assert seq_a.block_table.physical_blocks == []
    assert seq_b.block_table.physical_blocks != []


def test_handle_iteration_results_stamps_token_and_completion_times():
    scheduler = ContinuousBatchedScheduler()
    alloc = BlockAllocator(num_blocks=4, block_size=16)
    seq = make_sequence(status="DECODING")
    running = [seq]

    scheduler.handle_iteration_results([(1, 7, False)], running, alloc, loop=RecordingLoop())
    scheduler.handle_iteration_results([(1, 8, True)], running, alloc, loop=RecordingLoop())

    metrics = seq.metrics
    assert metrics.first_token_time == metrics.token_times[0]
    assert len(metrics.token_times) == 2
    assert metrics.done_time == metrics.token_times[-1]


def test_abort_frees_blocks_and_drops_sequences_from_running():
    scheduler = ContinuousBatchedScheduler()
    alloc = BlockAllocator(num_blocks=4, block_size=4)
    keep = decoding_sequence(alloc, seq_id=1, prompt_len=4)
    drop = decoding_sequence(alloc, seq_id=2, prompt_len=4)
    running = [keep, drop]

    scheduler.abort([drop], running, alloc, loop=RecordingLoop())

    assert running == [keep]
    assert len(alloc.free_blocks) == 3


def test_continuous_scheduler_admits_while_others_are_running():
    alloc = BlockAllocator(num_blocks=100, block_size=16)
    decoding = decoding_sequence(alloc, seq_id=1, prompt_len=5)
    newcomer = make_sequence(seq_id=2)
    running, waiting = [decoding], deque([newcomer])

    plan = ContinuousBatchedScheduler().scheduler_step(running, waiting, alloc)

    assert newcomer in running
    assert {e.seq_id for e in plan} == {1, 2}


def test_continuous_scheduler_admission_stops_at_the_concurrency_cap():
    alloc = BlockAllocator(num_blocks=100, block_size=16)
    decoding = decoding_sequence(alloc, seq_id=1, prompt_len=5)
    newcomer = make_sequence(seq_id=2)
    running, waiting = [decoding], deque([newcomer])

    plan = ContinuousBatchedScheduler(max_concurrent_sequences=1).scheduler_step(
        running, waiting, alloc
    )

    assert list(waiting) == [newcomer]
    assert {e.seq_id for e in plan} == {1}


def test_static_scheduler_admits_nothing_until_the_running_batch_drains():
    alloc = BlockAllocator(num_blocks=100, block_size=16)
    decoding = decoding_sequence(alloc, seq_id=1, prompt_len=5)
    newcomer = make_sequence(seq_id=2)
    running, waiting = [decoding], deque([newcomer])

    plan = StaticBatchedScheduler().scheduler_step(running, waiting, alloc)

    assert list(waiting) == [newcomer]
    assert {e.seq_id for e in plan} == {1}


def test_static_scheduler_admits_up_to_batch_size_once_running_is_empty():
    alloc = BlockAllocator(num_blocks=100, block_size=16)
    seqs = [make_sequence(seq_id=i) for i in (1, 2, 3)]
    running, waiting = [], deque(seqs)

    plan = StaticBatchedScheduler(batch_size=2).scheduler_step(running, waiting, alloc)

    assert running == seqs[:2]
    assert list(waiting) == seqs[2:]
    assert {e.seq_id for e in plan} == {1, 2}
