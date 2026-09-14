"""Scheduler interface and implementations.

`Scheduler` is the contract the engine drives each iteration: build one BatchPlan from
current state (`scheduler_step`), then fold the GPU worker's results back into that
state (`handle_iteration_results`). Implementations own the admission policy; result
handling — free blocks, drop finished sequences — is the same regardless of policy and
lives on the base class.

`ContinuousBatchedScheduler` builds one iteration's BatchPlan in priority order:

  1. Already-DECODING sequences first (one token each) — a client mid-stream is never
     stalled to make room for something else.
  2. PREFILLING sequences continue their chunk where they left off.
  3. New admissions from `waiting`, gated by remaining budget, block-allocator
     capacity, and `max_concurrent_sequences`; if the head of the queue doesn't fit,
     admission stops rather than skipping ahead to a smaller request behind it.

TOKEN_BUDGET bounds how much compute one iteration spends on prefill, so a large
prompt can't spike inter-token latency for sequences decoding alongside it.
MAX_CONCURRENT_SEQUENCES bounds how many sequences share one iteration's decode step:
every running sequence gets a token every iteration regardless of how many there are,
so an unbounded `running` makes each iteration's — and so every sequence's inter-token
latency — grow with backlog size. Past this cap, excess demand queues in `waiting`
(schedule_latency) instead of degrading decode throughput for sequences already admitted.

`StaticBatchedScheduler` admits up to `batch_size` requests together and blocks all
further admission until every sequence in that batch has finished — a different
latency/throughput tradeoff than continuous batching, with no chunked prefill or
per-token admission decisions.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections import deque

from .allocator import BlockAllocator
from .batch_plan import BatchPlan
from .dispatch import dispatch_results
from .sequence import Sequence

TOKEN_BUDGET = 4096
BATCH_SIZE = 8
MAX_CONCURRENT_SEQUENCES = 64


class Scheduler(ABC):
    """One iteration's policy: what runs (`scheduler_step`) and how results feed back
    into engine state (`handle_iteration_results`)."""

    @abstractmethod
    def scheduler_step(
        self, running: list[Sequence], waiting: deque[Sequence], allocator: BlockAllocator
    ) -> BatchPlan: ...

    def handle_iteration_results(
        self,
        iter_results: list[tuple[int, int, bool]],
        running: list[Sequence],
        allocator: BlockAllocator,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        id_to_seq = {seq.seq_id: seq for seq in running}
        for seq_id, token, finished in iter_results:
            seq = id_to_seq[seq_id]
            seq.generated_tokens.append(token)
            if finished:
                allocator.free(seq.block_table)
                running.remove(seq)
        dispatch_results(iter_results, loop)


class ContinuousBatchedScheduler(Scheduler):
    def __init__(
        self,
        token_budget: int = TOKEN_BUDGET,
        max_concurrent_sequences: int = MAX_CONCURRENT_SEQUENCES,
    ) -> None:
        self.token_budget = token_budget
        self.max_concurrent_sequences = max_concurrent_sequences

    def scheduler_step(
        self, running: list[Sequence], waiting: deque[Sequence], allocator: BlockAllocator
    ) -> BatchPlan:
        budget = self.token_budget
        plan = BatchPlan()

        for seq in running:
            if seq.status == "DECODING":
                success: bool = allocator.get_capacity(seq, new_tokens=1)
                assert success, "could not allocate more blocks in decode"
                plan.add(seq, n_tokens=1, is_prefill_chunk=False)

        for seq in running:
            if seq.status == "PREFILLING" and budget > 0:
                remaining = len(seq.prompt_tokens) - seq.prefill_progress
                chunk = min(remaining, budget)
                success: bool = allocator.get_capacity(seq, new_tokens=chunk)
                assert success, "could not allocate more blocks in prefill"
                plan.add(seq, n_tokens=chunk, is_prefill_chunk=True)
                budget -= chunk
                seq.prefill_progress += chunk
                if seq.prefill_progress == len(seq.prompt_tokens):
                    seq.status = "DECODING"

        while waiting and budget > 0 and len(running) < self.max_concurrent_sequences:
            seq = waiting[0]
            chunk = min(len(seq.prompt_tokens), budget)
            success: bool = allocator.get_capacity(seq, chunk)
            if not success:
                break
            plan.add(seq, n_tokens=chunk, is_prefill_chunk=True)
            budget -= chunk
            seq.prefill_progress += chunk
            if seq.prefill_progress == len(seq.prompt_tokens):
                seq.status = "DECODING"
            else:
                seq.status = "PREFILLING"
            running.append(waiting.popleft())

        return plan


class StaticBatchedScheduler(Scheduler):
    """Admits up to `batch_size` requests together and blocks all further admission
    until every sequence in that batch has finished."""

    def __init__(self, batch_size: int = BATCH_SIZE, token_budget: int = TOKEN_BUDGET) -> None:
        self.batch_size = batch_size
        self.token_budget = token_budget

    def scheduler_step(
        self, running: list[Sequence], waiting: deque[Sequence], allocator: BlockAllocator
    ) -> BatchPlan:
        budget = self.token_budget
        plan = BatchPlan()

        for seq in running:
            if seq.status == "DECODING":
                success: bool = allocator.get_capacity(seq, new_tokens=1)
                assert success, "could not allocate more blocks in decode"
                plan.add(seq, n_tokens=1, is_prefill_chunk=False)

        for seq in running:
            if seq.status == "PREFILLING" and budget > 0:
                remaining = len(seq.prompt_tokens) - seq.prefill_progress
                chunk = min(remaining, budget)
                success: bool = allocator.get_capacity(seq, new_tokens=chunk)
                assert success, "could not allocate more blocks in prefill"
                plan.add(seq, n_tokens=chunk, is_prefill_chunk=True)
                budget -= chunk
                seq.prefill_progress += chunk
                if seq.prefill_progress == len(seq.prompt_tokens):
                    seq.status = "DECODING"

        if running:
            return plan  # batch still in flight — next batch can't start admitting yet

        while waiting and len(running) < self.batch_size and budget > 0:
            seq = waiting[0]
            chunk = min(len(seq.prompt_tokens), budget)
            success: bool = allocator.get_capacity(seq, chunk)
            if not success:
                break
            plan.add(seq, n_tokens=chunk, is_prefill_chunk=True)
            budget -= chunk
            seq.prefill_progress += chunk
            if seq.prefill_progress == len(seq.prompt_tokens):
                seq.status = "DECODING"
            else:
                seq.status = "PREFILLING"
            running.append(waiting.popleft())

        return plan
