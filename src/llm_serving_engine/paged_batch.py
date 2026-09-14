"""Per-iteration paging metadata shared by both forward paths that read the paged KV
pool (ModelRunner.forward_fused and DecodeGraphRunner)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass(slots=True)
class PagedBatch:
    """Computed once per iteration and shared across every layer's attention_forward
    call -- depends only on the plan and each Sequence's BlockTable, not on any layer's
    K/V. One row per plan entry, in plan order."""

    block_table: torch.Tensor  # (num_entries, max_blocks), physical block ids
    context_len: torch.Tensor  # (num_entries,), total cached length after this call's write
    query_offset: torch.Tensor  # (num_entries,), cached length before this call's new tokens
    q_start: torch.Tensor  # (num_entries,), entry's start offset in the flattened q/out
    q_len: torch.Tensor  # (num_entries,), entry's token count this call
    max_q_len: int  # q_len's max
    dest_block_id: torch.Tensor  # (total_tokens,), physical block per new K/V row
    dest_within: torch.Tensor  # (total_tokens,), within-block offset per new K/V row
