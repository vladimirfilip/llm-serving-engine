"""One iteration's work, built by `scheduler_step` on the scheduler thread and consumed by
`ModelRunner.forward` on the GPU worker thread."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from .sequence import Sequence


@dataclass(slots=True)
class BatchEntry:
    seq_id: int
    n_tokens: int
    is_prefill_chunk: bool = False


@dataclass(slots=True)
class BatchPlan:
    entries: list[BatchEntry] = field(default_factory=list)
    # Sequences sent back to `waiting` this step; their per-sequence model state must be freed.
    preempted: list[int] = field(default_factory=list)
    # Sequences the KV pool can never hold; their streams must end with an error.
    rejected: list[Sequence] = field(default_factory=list)

    def add(self, seq: Sequence, n_tokens: int, is_prefill_chunk: bool = False) -> None:
        self.entries.append(BatchEntry(seq.seq_id, n_tokens, is_prefill_chunk))

    def __iter__(self) -> Iterator[BatchEntry]:
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)
