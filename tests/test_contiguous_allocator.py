"""ContiguousAllocator correctness: whole-sequence reservation sized to prompt +
max_tokens, charged once at admission and returned whole on free(). Assert behaviour
and invariants, not implementation detail.
"""

from llm_serving_engine.model.sampling import SamplingParams
from llm_serving_engine.scheduling.allocator import ContiguousAllocator
from tests.test_sequence import make_sequence


def test_first_call_reserves_prompt_plus_max_tokens():
    alloc = ContiguousAllocator(capacity_tokens=100)
    seq = make_sequence(prompt_tokens=[1, 2, 3], sampling_params=SamplingParams(max_tokens=10))

    assert alloc.get_capacity(seq, new_tokens=3)  # prefill: the 3 prompt tokens
    assert seq.block_table.reserved_tokens == 13  # 3 prompt + 10 max_tokens, not just new_tokens
    assert seq.block_table.num_tokens == 3
    assert alloc.used_tokens == 13


def test_later_calls_grow_usage_without_charging_a_second_reservation():
    alloc = ContiguousAllocator(capacity_tokens=100)
    seq = make_sequence(prompt_tokens=[1, 2, 3], sampling_params=SamplingParams(max_tokens=10))

    alloc.get_capacity(seq, new_tokens=3)
    alloc.get_capacity(seq, new_tokens=1)  # one decode step

    assert seq.block_table.num_tokens == 4
    assert seq.block_table.reserved_tokens == 13
    assert alloc.used_tokens == 13  # unchanged: reservation was already worst-case


def test_get_capacity_returns_false_and_does_not_mutate_when_pool_is_short():
    alloc = ContiguousAllocator(capacity_tokens=10)
    seq = make_sequence(prompt_tokens=[1, 2, 3], sampling_params=SamplingParams(max_tokens=10))

    assert not alloc.get_capacity(seq, new_tokens=3)  # needs 13, pool has 10
    assert seq.block_table.reserved_tokens == 0
    assert seq.block_table.num_tokens == 0
    assert alloc.used_tokens == 0


def test_two_sequences_share_capacity_by_worst_case_not_actual_usage():
    alloc = ContiguousAllocator(capacity_tokens=20)
    a = make_sequence(seq_id=1, prompt_tokens=[1], sampling_params=SamplingParams(max_tokens=9))
    b = make_sequence(seq_id=2, prompt_tokens=[1], sampling_params=SamplingParams(max_tokens=9))

    assert alloc.get_capacity(a, new_tokens=1)  # reserves 10
    assert alloc.get_capacity(b, new_tokens=1)  # reserves 10, exactly fills the pool
    assert alloc.used_tokens == 20


def test_free_returns_the_whole_reservation_and_clears_table():
    alloc = ContiguousAllocator(capacity_tokens=100)
    seq = make_sequence(prompt_tokens=[1, 2, 3], sampling_params=SamplingParams(max_tokens=10))
    alloc.get_capacity(seq, new_tokens=3)

    alloc.free(seq.block_table)  # freed after generating only 2 tokens, far short of max_tokens

    assert alloc.used_tokens == 0
    assert seq.block_table.reserved_tokens == 0
    assert seq.block_table.num_tokens == 0
