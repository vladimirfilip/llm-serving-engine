"""KV-cache allocators. Both expose `allocate`, `free`, `can_ever_fit` and `utilization`,
so the scheduler drives either one.

BlockAllocator grows a sequence's block table one block at a time as it needs slots.
ContiguousAllocator reserves prompt + max_tokens at admission as one slab.
"""

from __future__ import annotations

from collections import deque
from math import ceil

from .sequence import BlockTable, Sequence


class BlockAllocator:
    def __init__(self, num_blocks: int, block_size: int):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.free_blocks: deque[int] = deque(range(num_blocks))

    def needed_blocks(self, table: BlockTable, new_tokens: int) -> int:
        return ceil((table.num_tokens + new_tokens) / self.block_size) - len(table.physical_blocks)

    def allocate(self, seq: Sequence, new_tokens: int, decode_reserve: int = 0) -> bool:
        """Grows `seq` by `new_tokens` slots, or returns False and changes nothing.
        `decode_reserve` blocks stay free afterwards: each running sequence may need one
        new block for its next decode token."""
        needed = self.needed_blocks(seq.block_table, new_tokens)
        if needed + decode_reserve > len(self.free_blocks):
            return False
        seq.block_table.num_tokens += new_tokens
        for _ in range(needed):
            seq.block_table.physical_blocks.append(self.free_blocks.popleft())
        return True

    def free(self, table: BlockTable) -> None:
        self.free_blocks.extend(table.physical_blocks)
        table.physical_blocks.clear()
        table.num_tokens = 0

    def can_ever_fit(self, seq: Sequence) -> bool:
        """Whether the whole pool could hold `seq` plus its next decode token."""
        return ceil((seq.num_tokens + 1) / self.block_size) <= self.num_blocks

    @property
    def utilization(self) -> float:
        return 1 - len(self.free_blocks) / self.num_blocks if self.num_blocks else 0.0


class ContiguousAllocator:
    """A sequence that finishes far short of max_tokens still holds its whole reservation
    until freed. Give it `capacity_tokens = num_blocks * block_size` of the paged pool it
    is compared against, so both run at matched memory."""

    def __init__(self, capacity_tokens: int):
        self.capacity_tokens = capacity_tokens
        self.used_tokens = 0

    def allocate(self, seq: Sequence, new_tokens: int, decode_reserve: int = 0) -> bool:
        """Decode growth always lands inside the sequence's own reservation, so
        `decode_reserve` never applies here."""
        table = seq.block_table
        if table.reserved_tokens == 0:
            reservation = self._reservation(seq)
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

    def can_ever_fit(self, seq: Sequence) -> bool:
        return self._reservation(seq) <= self.capacity_tokens

    @property
    def utilization(self) -> float:
        return self.used_tokens / self.capacity_tokens if self.capacity_tokens else 0.0

    @staticmethod
    def _reservation(seq: Sequence) -> int:
        return len(seq.prompt_tokens) + seq.sampling_params.max_tokens


KVAllocator = BlockAllocator | ContiguousAllocator
