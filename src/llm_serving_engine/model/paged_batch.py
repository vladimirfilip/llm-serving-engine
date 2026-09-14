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


def pinned(rows: list[list[int]], dtype: torch.dtype) -> torch.Tensor:
    """Page-locked host tensor, so a `non_blocking` copy to the device neither blocks the
    host nor goes through a pageable staging buffer."""
    return torch.tensor(rows, dtype=dtype).pin_memory()
