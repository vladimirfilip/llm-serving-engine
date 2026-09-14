"""KV-cache allocators: BlockAllocator (paged, grows a sequence's block table
incrementally) and ContiguousAllocator (a naive baseline that reserves each
sequence's worst-case span up front).

Pool sizing (bytes/token/layer = 2 * n_kv_heads * head_dim * dtype_bytes) uses
n_kv_heads, not n_heads, since K/V projections are shared across a group of query
heads under grouped-query attention.

Both expose the same `get_capacity(seq, new_tokens) -> bool` / `free(table) -> None`
surface the scheduler calls, so either can be dropped into InferenceEngine unmodified.
"""

from __future__ import annotations

from collections import deque
from math import ceil

from .sequence import BlockTable, Sequence


class BlockAllocator:
    def __init__(self, num_blocks: int, block_size: int):
        self.free_blocks: deque[int] = deque(range(num_blocks))
        self.block_size = block_size

    def needed_blocks(self, table: BlockTable, new_tokens: int) -> int:
        return ceil((table.num_tokens + new_tokens) / self.block_size) - len(table.physical_blocks)

    def get_capacity(self, seq: Sequence, new_tokens: int) -> bool:
        needed = self.needed_blocks(seq.block_table, new_tokens)
        if needed > len(self.free_blocks):
            return False
        seq.block_table.num_tokens += new_tokens
        for _ in range(needed):
            seq.block_table.physical_blocks.append(self.free_blocks.popleft())
        return True

    def free(self, table: BlockTable) -> None:
        self.free_blocks.extend(table.physical_blocks)
        table.physical_blocks.clear()


class ContiguousAllocator:
    """Naive per-sequence max-length reservation: reserves prompt + max_tokens worth of
    capacity as one slab at admission, not incrementally in blocks, so a sequence that
    finishes far short of max_tokens still holds its worst-case share until it's freed.
    Construct it with `capacity_tokens` equal to `num_blocks * block_size` of a paged
    pool it's being compared against, so the two run at matched memory.
    """

    def __init__(self, capacity_tokens: int):
        self.capacity_tokens = capacity_tokens
        self.used_tokens = 0

    def get_capacity(self, seq: Sequence, new_tokens: int) -> bool:
        table = seq.block_table
        if table.reserved_tokens == 0:
            reservation = len(seq.prompt_tokens) + seq.sampling_params.max_tokens
            if self.used_tokens + reservation > self.capacity_tokens:
                return False
            table.reserved_tokens = reservation
            self.used_tokens += reservation
        table.num_tokens += new_tokens
        return True

    def free(self, table: BlockTable) -> None:
        self.used_tokens -= table.reserved_tokens
        table.reserved_tokens = 0
        table.num_tokens = 0
