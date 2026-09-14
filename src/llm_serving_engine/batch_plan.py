"""The data contract crossing the scheduler -> GPU worker boundary.

Built by scheduler_step and consumed by the model runner's tensor-building glue
(build_tensors). One BatchPlan crosses per iteration, carrying every sequence's work
for that iteration, rather than one call per sequence.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class BatchEntry:
    seq_id: int
    n_tokens: int
    is_prefill_chunk: bool = False


@dataclass
class BatchPlan:
    entries: list[BatchEntry] = field(default_factory=list)

    def add(self, seq, n_tokens: int, is_prefill_chunk: bool = False) -> None:
        self.entries.append(BatchEntry(seq.seq_id, n_tokens, is_prefill_chunk))

    def __iter__(self):
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def total_tokens(self) -> int:
        return sum(e.n_tokens for e in self.entries)
