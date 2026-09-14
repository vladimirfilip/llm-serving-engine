"""BlockAllocator correctness: block sizing via ceil-division, no mutation on a failed
check, and blocks returning to the free pool on free(). Assert behaviour and
invariants, not the exact free-list ordering.
"""

from collections import deque

from llm_serving_engine.scheduling.allocator import BlockAllocator
from llm_serving_engine.scheduling.sequence import BlockTable


def test_init_builds_free_list():
    alloc = BlockAllocator(num_blocks=4, block_size=16)
    assert alloc.free_blocks == deque([0, 1, 2, 3])
    assert alloc.block_size == 16


def test_get_capacity_allocates_exactly_the_blocks_a_fresh_table_needs():
    alloc = BlockAllocator(num_blocks=4, block_size=16)
    table = BlockTable()

    class _Seq:
        block_table = table

    assert alloc.get_capacity(_Seq(), new_tokens=17)  # ceil(17/16) = 2 blocks
    assert table.num_tokens == 17
    assert len(table.physical_blocks) == 2
    assert len(alloc.free_blocks) == 2


def test_get_capacity_only_grows_by_the_shortfall_on_a_later_call():
    alloc = BlockAllocator(num_blocks=4, block_size=16)
    table = BlockTable()

    class _Seq:
        block_table = table

    seq = _Seq()
    alloc.get_capacity(seq, new_tokens=16)  # exactly fills block 0
    assert len(table.physical_blocks) == 1

    alloc.get_capacity(seq, new_tokens=1)  # tips into block 1
    assert table.num_tokens == 17
    assert len(table.physical_blocks) == 2
    assert len(alloc.free_blocks) == 2


def test_get_capacity_returns_false_and_does_not_mutate_when_pool_is_short():
    alloc = BlockAllocator(num_blocks=1, block_size=16)
    table = BlockTable()

    class _Seq:
        block_table = table

    assert not alloc.get_capacity(_Seq(), new_tokens=32)  # needs 2 blocks, pool has 1
    assert table.num_tokens == 0
    assert table.physical_blocks == []
    assert len(alloc.free_blocks) == 1


def test_free_returns_blocks_and_clears_table():
    alloc = BlockAllocator(num_blocks=4, block_size=16)
    table = BlockTable(physical_blocks=[0, 1], num_tokens=17)
    alloc.free_blocks = deque([2, 3])

    alloc.free(table)

    assert sorted(alloc.free_blocks) == [0, 1, 2, 3]
    assert table.physical_blocks == []
