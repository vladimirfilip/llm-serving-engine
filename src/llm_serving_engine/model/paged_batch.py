"""Per-iteration paging metadata read by every layer's paged attention, and the host-side
staging that ships it to the device."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class PagedBatch:
    """One row per plan entry, in plan order; identical for every layer."""

    block_table: torch.Tensor  # (num_entries, max_blocks) int32, physical block ids
    context_len: torch.Tensor  # (num_entries,) int32, cached length after this call's write
    query_offset: torch.Tensor  # (num_entries,) int32, cached length before this call's tokens
    q_start: torch.Tensor  # (num_entries,) int32, entry's first row in the flattened q/out
    q_len: torch.Tensor  # (num_entries,) int32, entry's token count this call
    max_q_len: int
    dest_block_id: torch.Tensor  # (total_tokens,) int64, physical block per new K/V row
    dest_within: torch.Tensor  # (total_tokens,) int64, offset within that block


@dataclass(slots=True)
class PagingRows:
    """PagedBatch's fields as host lists, before they reach the device."""

    block_table: list[list[int]]
    context_len: list[int]
    query_offset: list[int]
    q_start: list[int]
    q_len: list[int]
    dest_block_id: list[int]
    dest_within: list[int]

    def to_device(self, device: str) -> PagedBatch:
        """Three packed host-to-device copies for the whole batch."""
        per_entry = pinned(
            [self.context_len, self.query_offset, self.q_start, self.q_len], torch.int32
        ).to(device, non_blocking=True)
        dest = pinned([self.dest_block_id, self.dest_within], torch.int64)
        dest = dest.to(device, non_blocking=True)
        return PagedBatch(
            block_table=pinned(self.block_table, torch.int32).to(device, non_blocking=True),
            context_len=per_entry[0],
            query_offset=per_entry[1],
            q_start=per_entry[2],
            q_len=per_entry[3],
            max_q_len=max(self.q_len),
            dest_block_id=dest[0],
            dest_within=dest[1],
        )


def pinned(rows: list[list[int]], dtype: torch.dtype) -> torch.Tensor:
    """Page-locked host tensor, so a `non_blocking` copy to the device neither blocks the
    host nor goes through a pageable staging buffer."""
    return torch.tensor(rows, dtype=dtype).pin_memory()
