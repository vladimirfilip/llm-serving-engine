"""Request-lifetime state. Plain data: the scheduler and allocator own every mutation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ..model.sampling import SamplingParams
from ..observability.metrics import RequestMetrics

SequenceStatus = Literal["WAITING", "PREFILLING", "DECODING"]


@dataclass(slots=True)
class BlockTable:
    physical_blocks: list[int] = field(default_factory=list)
    num_tokens: int = 0  # tokens with a KV slot, including any this iteration writes
    reserved_tokens: int = 0  # ContiguousAllocator's whole-lifetime reservation


@dataclass(slots=True, eq=False)
class Sequence:
    seq_id: int
    prompt_tokens: list[int]
    sampling_params: SamplingParams
    metrics: RequestMetrics
    generated_tokens: list[int] = field(default_factory=list)
    block_table: BlockTable = field(default_factory=BlockTable)
    status: SequenceStatus = "WAITING"
    prefill_progress: int = 0

    @property
    def num_tokens(self) -> int:
        """Prompt plus generated; also the prefill length of a WAITING sequence, since a
        preempted sequence recomputes the KV of every token it already generated."""
        return len(self.prompt_tokens) + len(self.generated_tokens)

    def prefill_token_ids(self, start: int, end: int) -> list[int]:
        if end <= len(self.prompt_tokens):
            return self.prompt_tokens[start:end]
        return (self.prompt_tokens + self.generated_tokens)[start:end]
