"""BlockAllocator: ceil-division block sizing, no mutation when an allocation fails, and
blocks returning to the free pool on free()."""

from llm_serving_engine.scheduling.allocator import BlockAllocator
from tests.factories import make_sequence


def test_allocate_takes_exactly_the_blocks_a_fresh_table_needs():
    alloc = BlockAllocator(num_blocks=4, block_size=16)
    seq = make_sequence()

    assert alloc.allocate(seq, new_tokens=17)  # ceil(17/16) = 2 blocks
    assert seq.block_table.num_tokens == 17
    assert len(seq.block_table.physical_blocks) == 2
    assert len(alloc.free_blocks) == 2


def test_allocate_grows_only_by_the_shortfall_on_a_later_call():
    alloc = BlockAllocator(num_blocks=4, block_size=16)
    seq = make_sequence()
    alloc.allocate(seq, new_tokens=16)  # fills block 0 exactly
    assert len(seq.block_table.physical_blocks) == 1

    alloc.allocate(seq, new_tokens=1)  # tips into block 1
    assert seq.block_table.num_tokens == 17
    assert len(seq.block_table.physical_blocks) == 2


def test_failed_allocate_changes_nothing():
    alloc = BlockAllocator(num_blocks=1, block_size=16)
    seq = make_sequence()

    assert not alloc.allocate(seq, new_tokens=32)  # needs 2 blocks, pool has 1
    assert seq.block_table.num_tokens == 0
    assert seq.block_table.physical_blocks == []
    assert len(alloc.free_blocks) == 1


def test_decode_reserve_keeps_blocks_free_after_the_allocation():
    alloc = BlockAllocator(num_blocks=3, block_size=4)
    seq = make_sequence()
    assert not alloc.allocate(seq, new_tokens=8, decode_reserve=2)  # 2 blocks + 2 reserved > 3
    assert alloc.allocate(seq, new_tokens=8, decode_reserve=1)


def test_free_returns_every_block_and_empties_the_table():
    alloc = BlockAllocator(num_blocks=4, block_size=16)
    seq = make_sequence()
    alloc.allocate(seq, new_tokens=17)

    alloc.free(seq.block_table)

    assert sorted(alloc.free_blocks) == [0, 1, 2, 3]
    assert seq.block_table.physical_blocks == []
    assert seq.block_table.num_tokens == 0


def test_can_ever_fit_counts_the_next_decode_token_against_the_whole_pool():
    alloc = BlockAllocator(num_blocks=2, block_size=4)
    assert alloc.can_ever_fit(make_sequence(prompt_tokens=[0] * 7))  # 8 slots, 2 blocks
    assert not alloc.can_ever_fit(make_sequence(prompt_tokens=[0] * 8))  # 9 slots, 3 blocks


def test_utilization_is_the_fraction_of_blocks_handed_out():
    alloc = BlockAllocator(num_blocks=10, block_size=4)
    alloc.allocate(make_sequence(), new_tokens=16)
    assert alloc.utilization == 0.4
