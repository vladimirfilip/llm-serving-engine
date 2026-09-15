"""CUDA graph capture and replay for pure-decode iterations.

One graph per batch-size bucket turns a decode step's eager per-layer kernel launches into
a single replay. ModelRunner.forward replays a graph only for a plan whose every entry is a
one-token decode and whose size fits a captured bucket.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING

import torch

from .paged_batch import PagedBatch, pinned

if TYPE_CHECKING:
    from ..scheduling.batch_plan import BatchPlan
    from ..scheduling.sequence import Sequence
    from .model_runner import IterationResults, ModelRunner

logger = logging.getLogger(__name__)


def decode_graph_buckets(max_batch: int) -> list[int]:
    """Powers of two up to the first that holds `max_batch`, so padding stays under 2x at
    any batch size the scheduler can run."""
    buckets = [1]
    while buckets[-1] < max_batch:
        buckets.append(buckets[-1] * 2)
    return buckets


@dataclass(slots=True)
class DecodeGraph:
    """A captured graph and the fixed-address tensors it reads and writes. A replay uses
    whatever these addresses hold at replay time, so each replay refills every row."""

    # (4, bucket) int64: input_ids, position_ids, dest_block_id, dest_within
    index_rows: torch.Tensor
    # (3, bucket) int32: context_len, query_offset, q_len (0 marks a padding row)
    length_rows: torch.Tensor
    q_start: torch.Tensor  # (bucket,) int32 arange: each row packs one token, at flat position i
    block_table: torch.Tensor  # (bucket, num_blocks) int32
    logits: torch.Tensor  # (bucket, vocab_size), written by the captured region
    graph: torch.cuda.CUDAGraph = field(default_factory=torch.cuda.CUDAGraph)
    # Per row: the seq_id whose block ids that block_table row holds, and how many of them.
    # Block tables only grow until the sequence is freed, so a row still holding the same
    # seq_id needs only its new trailing cells written.
    row_seq_ids: list[int | None] = field(default_factory=list)
    row_blocks_written: list[int] = field(default_factory=list)

    def paging(self) -> PagedBatch:
        return PagedBatch(
            block_table=self.block_table,
            context_len=self.length_rows[0],
            query_offset=self.length_rows[1],
            q_start=self.q_start,
            q_len=self.length_rows[2],
            max_q_len=1,
            dest_block_id=self.index_rows[2],
            dest_within=self.index_rows[3],
        )


class DecodeGraphRunner:
    """Captures, self-checks and replays one ModelRunner's decode graphs. The runner
    supplies the compute (decoder_hidden, paged_attention, emit_tokens) and the pool."""

    def __init__(self, runner: ModelRunner) -> None:
        self._runner = runner
        self.graphs: dict[int, DecodeGraph] = {}

    def bucket_for(self, n: int) -> int | None:
        """Smallest captured bucket holding a decode batch of n, or None."""
        return min((b for b in self.graphs if b >= n), default=None)

    def forget(self, seq_id: int) -> None:
        """A freed sequence's block ids are stale in every row that still records them."""
        for dg in self.graphs.values():
            for row, occupant in enumerate(dg.row_seq_ids):
                if occupant == seq_id:
                    dg.row_seq_ids[row] = None

    def capture(self, bucket_sizes: list[int]) -> None:
        """No-op without a KV pool. Running out of GPU memory, or a failed self-check,
        leaves no graphs, and decode runs eagerly on forward_fused."""
        runner = self._runner
        if runner._k_pool is None:
            return

        memory_pool = None
        try:
            for bucket in sorted(bucket_sizes):
                dg = self._alloc(bucket, num_blocks=runner._scratch_block_id)
                self._fill_dummy(dg)
                warmup = runner.graph_warmup_stream()
                warmup.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(warmup):
                    for _ in range(3):
                        self._step_core(dg)
                torch.cuda.current_stream().wait_stream(warmup)
                with torch.cuda.graph(dg.graph, pool=memory_pool):
                    self._step_core(dg)
                memory_pool = dg.graph.pool()
                self.graphs[bucket] = dg
        except torch.cuda.OutOfMemoryError:
            logger.exception("CUDA graph capture ran out of memory; decode runs eagerly")
            self.graphs.clear()
            return

        if not self._self_check():
            logger.error("a captured decode graph disagrees with eager decode; decode runs eagerly")
            self.graphs.clear()

    def replay(self, plan: BatchPlan, seqs: dict[int, Sequence], bucket: int) -> IterationResults:
        dg = self.graphs[bucket]
        owed = [seqs[entry.seq_id] for entry in plan]
        self._fill_real(dg, owed)
        dg.graph.replay()
        return self._runner.emit_tokens(dg.logits, owed)

    def _step_core(self, dg: DecodeGraph) -> None:
        """Embedding through lm_head into dg.logits: the region capture() records, also run
        eagerly for warmup and the self-check.

        Sampling stays outside the graph. sample_tokens branches in Python on the batch's
        SamplingParams, and a graph freezes which kernels ran, so a captured sampler would
        replay one batch's sampling algorithm onto whichever requests fill the rows later.
        """
        runner = self._runner
        with torch.no_grad():
            hidden = runner.decoder_hidden(
                dg.index_rows[0:1],
                dg.index_rows[1:2],
                partial(runner.paged_attention, paging=dg.paging()),
            )
            dg.logits.copy_(runner.model.get_output_embeddings()(hidden[0]))

    def _alloc(self, bucket: int, num_blocks: int) -> DecodeGraph:
        runner = self._runner
        d = runner.device
        return DecodeGraph(
            index_rows=torch.zeros(4, bucket, dtype=torch.int64, device=d),
            length_rows=torch.zeros(3, bucket, dtype=torch.int32, device=d),
            q_start=torch.arange(bucket, dtype=torch.int32, device=d),
            block_table=torch.zeros(bucket, num_blocks, dtype=torch.int32, device=d),
            logits=torch.zeros(
                bucket, runner.model.config.vocab_size, dtype=runner.model.dtype, device=d
            ),
            row_seq_ids=[None] * bucket,
            row_blocks_written=[0] * bucket,
        )

    def _fill_dummy(self, dg: DecodeGraph) -> None:
        """Every row real (q_len=1) so the full compute path runs, and every read and write
        on the scratch block, which no sequence ever owns."""
        scratch = self._runner._scratch_block_id
        dg.index_rows.zero_()
        dg.index_rows[2].fill_(scratch)
        dg.length_rows[0].fill_(1)  # context_len
        dg.length_rows[1].zero_()  # query_offset
        dg.length_rows[2].fill_(1)  # q_len
        dg.block_table.fill_(scratch)
        dg.row_seq_ids[:] = [None] * len(dg.row_seq_ids)

    def _fill_real(self, dg: DecodeGraph, owed: list[Sequence]) -> None:
        """Rows past len(owed) are padding: q_len=0 makes the kernel skip them, and their
        unmasked K/V write lands on the scratch block."""
        runner = self._runner
        block_size, scratch = runner._block_size, runner._scratch_block_id
        bucket = len(dg.row_seq_ids)

        input_ids, position_ids = [0] * bucket, [0] * bucket
        dest_block_id, dest_within = [scratch] * bucket, [0] * bucket
        context_len, query_offset, q_len = [0] * bucket, [0] * bucket, [0] * bucket
        cell_rows: list[int] = []
        cell_cols: list[int] = []
        cell_block_ids: list[int] = []

        for i, seq in enumerate(owed):
            physical = seq.block_table.physical_blocks
            pos = seq.block_table.num_tokens - 1  # the slot this step's token writes
            input_ids[i] = seq.generated_tokens[-1]
            position_ids[i] = pos
            dest_block_id[i] = physical[pos // block_size]
            dest_within[i] = pos % block_size
            context_len[i] = pos + 1
            query_offset[i] = pos
            q_len[i] = 1

            written = dg.row_blocks_written[i] if dg.row_seq_ids[i] == seq.seq_id else 0
            for col in range(written, len(physical)):
                cell_rows.append(i)
                cell_cols.append(col)
                cell_block_ids.append(physical[col])
            dg.row_seq_ids[i] = seq.seq_id
            dg.row_blocks_written[i] = len(physical)

        dg.index_rows.copy_(
            pinned([input_ids, position_ids, dest_block_id, dest_within], torch.int64),
            non_blocking=True,
        )
        dg.length_rows.copy_(
            pinned([context_len, query_offset, q_len], torch.int32), non_blocking=True
        )
        if cell_rows:
            cells = pinned([cell_rows, cell_cols, cell_block_ids], torch.int64)
            cells = cells.to(runner.device, non_blocking=True)
            dg.block_table.index_put_((cells[0], cells[1]), cells[2].to(torch.int32))

    def _self_check(self) -> bool:
        """Compares each graph's replay against eager execution on the same input. A bad
        capture can fault with a CUDA error that surfaces on a later, unrelated call, so
        this runs once at startup, before any request reaches a graph."""
        for dg in self.graphs.values():
            self._fill_dummy(dg)
            self._step_core(dg)
            eager_logits = dg.logits.clone()
            dg.graph.replay()
            torch.cuda.synchronize()
            if not torch.allclose(dg.logits, eager_logits, atol=1e-2, rtol=1e-2):
                return False
        return True
