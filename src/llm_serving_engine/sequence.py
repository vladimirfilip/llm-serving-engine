"""Request-lifetime state.

These are plain data containers. The behavior that mutates them (admission, chunked
prefill, block allocation) belongs to the scheduler and allocator, not here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .metrics import RequestMetrics
from .sampling import SamplingParams

SequenceStatus = Literal["WAITING", "PREFILLING", "DECODING", "FINISHED"]


@dataclass(slots=True)
class BlockTable:
    physical_blocks: list[int] = field(default_factory=list)
    num_tokens: int = 0
    reserved_tokens: int = 0  # ContiguousAllocator's up-front reservation; unused by BlockAllocator


@dataclass(slots=True)
class Sequence:
    seq_id: int
    prompt_tokens: list[int]
    sampling_params: SamplingParams
    arrival_time: float
    metrics: RequestMetrics
    generated_tokens: list[int] = field(default_factory=list)
    block_table: BlockTable = field(default_factory=BlockTable)
    status: SequenceStatus = "WAITING"
    prefill_progress: int = 0

    @property
    def is_finished(self) -> bool:
        return self.status == "FINISHED"

    @property
    def num_tokens(self) -> int:
        return len(self.prompt_tokens) + len(self.generated_tokens)
