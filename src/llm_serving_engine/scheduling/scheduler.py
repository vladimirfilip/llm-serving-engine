"""Scheduler interface and implementations.

The engine drives a `Scheduler` once per iteration: `scheduler_step` builds one BatchPlan
from `running` and `waiting`, then `handle_iteration_results` folds the GPU worker's tokens
back in. Subclasses differ only in `admission_cap`.

`scheduler_step` plans in priority order:

  1. DECODING sequences, one token each. Decode tokens don't draw on TOKEN_BUDGET, so a
     client mid-stream is never stalled behind prefill work.
  2. The PREFILLING sequence continues its chunk, if its next chunk's blocks are free.
  3. Admissions from the head of `waiting`, gated by the remaining budget, the allocator
     and `admission_cap`. A head that doesn't fit stops admission; nothing skips it.

`running` stays in arrival order: admission only appends the head of `waiting`, and a
preempted sequence returns to the head. So `running[-1]` is always the latest arrival,
and when a decode token finds no free block, sequences are preempted from the tail. A
preempted sequence frees its blocks and later re-prefills prompt + generated tokens.

TOKEN_BUDGET bounds one iteration's prefill compute, so a long prompt can't spike
inter-token latency for the sequences decoding beside it. MAX_CONCURRENT_SEQUENCES bounds
the decode batch: every running sequence gets a token every iteration, so iteration time
grows with `running`; past the cap, demand waits in `waiting` as schedule latency.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections import deque

from .allocator import KVAllocator
from .batch_plan import BatchPlan
from .dispatch import dispatch_aborted, dispatch_results
from .sequence import Sequence

TOKEN_BUDGET = 4096
BATCH_SIZE = 8
MAX_CONCURRENT_SEQUENCES = 64


class Scheduler(ABC):
    def __init__(self, token_budget: int, max_running: int) -> None:
        self.token_budget = token_budget
        self.max_running = max_running

    @abstractmethod
    def admission_cap(self, running: list[Sequence]) -> int:
        """Most sequences `running` may hold once this step's admissions are done."""

    def scheduler_step(
        self, running: list[Sequence], waiting: deque[Sequence], allocator: KVAllocator
    ) -> BatchPlan:
        plan = BatchPlan()
        self._plan_decodes(running, waiting, allocator, plan)
        budget = self._plan_prefill_continuations(running, allocator, plan)
        self._plan_admissions(running, waiting, allocator, plan, budget)
        return plan

    def _plan_decodes(
        self, running: list[Sequence], waiting: deque[Sequence], allocator: KVAllocator,
        plan: BatchPlan,
    ) -> None:
        # Indexed: preemption pops the tail, which this loop hasn't reached yet.
        i = 0
        while i < len(running):
            seq = running[i]
            if seq.status == "DECODING" and _preempt_until_allocated(
                seq, running, waiting, allocator, plan
            ):
                plan.add(seq, n_tokens=1, is_prefill_chunk=False)
            i += 1

    def _plan_prefill_continuations(
        self, running: list[Sequence], allocator: KVAllocator, plan: BatchPlan
    ) -> int:
        """Returns the budget left for admissions. A chunk that doesn't fit waits and
        leaves no budget, so no later arrival takes the blocks it waits for; the
        PREFILLING sequence stays the tail, where decodes preempt first."""
        budget = self.token_budget
        for seq in running:
            if seq.status != "PREFILLING" or budget == 0:
                continue
            chunk = min(seq.num_tokens - seq.prefill_progress, budget)
            if not allocator.allocate(seq, chunk):
                return 0
            plan.add(seq, n_tokens=chunk, is_prefill_chunk=True)
            budget -= chunk
            _advance_prefill(seq, chunk)
        return budget

    def _plan_admissions(
        self, running: list[Sequence], waiting: deque[Sequence], allocator: KVAllocator,
        plan: BatchPlan, budget: int,
    ) -> None:
        cap = self.admission_cap(running)
        while waiting and budget > 0 and len(running) < cap:
            seq = waiting[0]
            if not allocator.can_ever_fit(seq):
                plan.rejected.append(waiting.popleft())
                continue
            chunk = min(seq.num_tokens, budget)
            # Headroom for every running sequence's next decode block; without it this
            # admission takes the last blocks and is itself preempted next step.
            if not allocator.allocate(seq, chunk, decode_reserve=len(running)):
                break
            plan.add(seq, n_tokens=chunk, is_prefill_chunk=True)
            budget -= chunk
            _advance_prefill(seq, chunk)
            if seq.metrics.admit_time is None:
                seq.metrics.admit_time = time.monotonic()
            running.append(waiting.popleft())

    def handle_iteration_results(
        self,
        iter_results: list[tuple[int, int, bool]],
        running: list[Sequence],
        allocator: KVAllocator,
        loop: asyncio.AbstractEventLoop,
    ) -> list[Sequence]:
        """Appends each token, frees finished sequences' blocks and drops them from
        `running`; returns the finished sequences."""
        now = time.monotonic()
        id_to_seq = {seq.seq_id: seq for seq in running}
        finished_seqs = []
        for seq_id, token, finished in iter_results:
            seq = id_to_seq[seq_id]
            seq.generated_tokens.append(token)
            metrics = seq.metrics
            if metrics.first_token_time is None:
                metrics.first_token_time = now
            metrics.token_times.append(now)
            if finished:
                metrics.done_time = now
                allocator.free(seq.block_table)
                running.remove(seq)
                finished_seqs.append(seq)
        dispatch_results(iter_results, loop)
        return finished_seqs

    def abort(
        self,
        seqs: list[Sequence],
        running: list[Sequence],
        allocator: KVAllocator,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        """Ends each sequence's stream with ABORTED and releases its blocks."""
        for seq in seqs:
            if seq in running:
                running.remove(seq)
            allocator.free(seq.block_table)
        dispatch_aborted([seq.seq_id for seq in seqs], loop)


class ContinuousBatchedScheduler(Scheduler):
    def __init__(
        self,
        token_budget: int = TOKEN_BUDGET,
        max_concurrent_sequences: int = MAX_CONCURRENT_SEQUENCES,
    ) -> None:
        super().__init__(token_budget, max_running=max_concurrent_sequences)

    def admission_cap(self, running: list[Sequence]) -> int:
        return self.max_running


class StaticBatchedScheduler(Scheduler):
    """Admits up to `batch_size` requests together, then admits nothing until every
    sequence in that batch has finished."""

    def __init__(self, batch_size: int = BATCH_SIZE, token_budget: int = TOKEN_BUDGET) -> None:
        super().__init__(token_budget, max_running=batch_size)

    def admission_cap(self, running: list[Sequence]) -> int:
        return 0 if running else self.max_running


def _advance_prefill(seq: Sequence, chunk: int) -> None:
    seq.prefill_progress += chunk
    seq.status = "DECODING" if seq.prefill_progress == seq.num_tokens else "PREFILLING"


def _preempt_until_allocated(
    seq: Sequence, running: list[Sequence], waiting: deque[Sequence], allocator: KVAllocator,
    plan: BatchPlan,
) -> bool:
    """Preempts from the tail of `running` until `seq` gets a slot for its next token.
    False if `seq` itself, as the tail, was preempted."""
    while not allocator.allocate(seq, new_tokens=1):
        victim = running.pop()
        allocator.free(victim.block_table)
        victim.prefill_progress = 0
        victim.status = "WAITING"
        waiting.appendleft(victim)
        plan.preempted.append(victim.seq_id)
        if victim is seq:
            return False
    return True
