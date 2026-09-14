"""ContiguousAllocator: one reservation of prompt + max_tokens, charged at admission and
returned whole on free()."""

from llm_serving_engine.model.sampling import SamplingParams
from llm_serving_engine.scheduling.allocator import ContiguousAllocator
from tests.factories import make_sequence


def _seq(**overrides):
    return make_sequence(
        prompt_tokens=[1, 2, 3], sampling_params=SamplingParams(max_tokens=10), **overrides
    )


def test_first_allocate_reserves_prompt_plus_max_tokens():
    alloc = ContiguousAllocator(capacity_tokens=100)
    seq = _seq()

    assert alloc.allocate(seq, new_tokens=3)
    assert seq.block_table.reserved_tokens == 13
    assert seq.block_table.num_tokens == 3
    assert alloc.used_tokens == 13


def test_later_allocations_draw_on_the_existing_reservation():
    alloc = ContiguousAllocator(capacity_tokens=100)
    seq = _seq()
    alloc.allocate(seq, new_tokens=3)
    alloc.allocate(seq, new_tokens=1)

    assert seq.block_table.num_tokens == 4
    assert alloc.used_tokens == 13


def test_failed_allocate_changes_nothing():
    alloc = ContiguousAllocator(capacity_tokens=10)
    seq = _seq()

    assert not alloc.allocate(seq, new_tokens=3)  # needs 13
    assert seq.block_table.reserved_tokens == 0
    assert seq.block_table.num_tokens == 0
    assert alloc.used_tokens == 0


def test_sequences_share_capacity_by_worst_case():
    alloc = ContiguousAllocator(capacity_tokens=20)
    a = make_sequence(seq_id=1, prompt_tokens=[1], sampling_params=SamplingParams(max_tokens=9))
    b = make_sequence(seq_id=2, prompt_tokens=[1], sampling_params=SamplingParams(max_tokens=9))

    assert alloc.allocate(a, new_tokens=1)
    assert alloc.allocate(b, new_tokens=1)
    assert alloc.used_tokens == 20
    assert alloc.utilization == 1.0


def test_free_returns_the_whole_reservation():
    alloc = ContiguousAllocator(capacity_tokens=100)
    seq = _seq()
    alloc.allocate(seq, new_tokens=3)

    alloc.free(seq.block_table)

    assert alloc.used_tokens == 0
    assert seq.block_table.reserved_tokens == 0
    assert seq.block_table.num_tokens == 0


def test_can_ever_fit_compares_the_reservation_with_capacity():
    assert ContiguousAllocator(capacity_tokens=13).can_ever_fit(_seq())
    assert not ContiguousAllocator(capacity_tokens=12).can_ever_fit(_seq())
