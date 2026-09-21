"""Piecewise CUDA graphs for iterations the full decode graphs can't take: any plan with a
prefill chunk, or a decode batch past the largest decode bucket.

Paged attention's launch grid depends on each iteration's entry count and longest chunk, so
it can't be captured. Everything between two attention calls is per-token work of fixed
shape once the token count is padded to a bucket, so each bucket captures L + 1 segments:

  segment 0      embedding, rotary tables, layer 0 up to its K/V write and q
  segment l      layer l-1 from o_proj through its MLP, then layer l up to its K/V write and q
  segment L      layer L-1 from o_proj through its MLP, then the final norm

and an iteration replays them with one eager attention launch between each pair, so Python
dispatches L attention launches per iteration where eager execution dispatches every kernel.

Segments hand tensors to each other, and to attention, through one set of buffers sized for
the largest bucket; a bucket's graphs read and write the leading `bucket` positions. Padding
positions carry token 0 at position 0 and write their K/V to the scratch block; attention
reads only real positions, and the per-token segments keep padding rows out of real ones.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from ..observability.nvtx import nvtx_range
from .paged_batch import PagedBatch, pinned

if TYPE_CHECKING:
    from ..scheduling.batch_plan import BatchPlan
    from ..scheduling.sequence import Sequence
    from .model_runner import IterationResults, ModelRunner

logger = logging.getLogger(__name__)


def piecewise_graph_buckets(max_tokens: int) -> list[int]:
    """Token counts from 16 up, each a power of two or 1.5 times one, ending at
    `max_tokens` itself: an iteration pads by under 1.5x past the smallest bucket."""
    candidates = sorted({2**i for i in range(4, 20)} | {3 * 2**i for i in range(3, 19)})
    return [size for size in candidates if size < max_tokens] + [max_tokens]


@dataclass(slots=True)
class SegmentBuffers:
    """Fixed-address tensors, sized for the largest bucket, that segments read and write."""

    # (4, max_tokens) int64: input_ids, position_ids, dest_block_id, dest_within
    index_rows: torch.Tensor
    cos: torch.Tensor  # (1, max_tokens, head_dim), rotary table for this iteration's positions
    sin: torch.Tensor  # (1, max_tokens, head_dim)
    residual: torch.Tensor  # (1, max_tokens, hidden_size), the current layer's input
    q: torch.Tensor  # (n_heads, max_tokens, head_dim), the current layer's rotated q
    attn: torch.Tensor  # (n_heads, max_tokens, head_dim), the current layer's attention output
    hidden: torch.Tensor  # (1, max_tokens, hidden_size), final-norm output


class PiecewiseGraphRunner:
    """Captures, self-checks and replays one ModelRunner's piecewise graphs."""

    def __init__(self, runner: ModelRunner) -> None:
        self._runner = runner
        self.graphs: dict[int, list[torch.cuda.CUDAGraph]] = {}
        self._buffers: SegmentBuffers | None = None

    def bucket_for(self, n_tokens: int) -> int | None:
        return min((b for b in self.graphs if b >= n_tokens), default=None)

    def capture(self, bucket_sizes: list[int]) -> None:
        """No-op without a KV pool. Running out of GPU memory, or a failed self-check,
        leaves no piecewise graphs, and those iterations run eagerly on forward_fused.
        Largest bucket first, so smaller buckets' captures reuse its memory."""
        runner = self._runner
        if runner._k_pool is None:
            return
        self._buffers = self._alloc_buffers(max(bucket_sizes))

        memory_pool = None
        try:
            for bucket in sorted(bucket_sizes, reverse=True):
                paging = self._fill_dummy(bucket)
                warmup = runner.graph_warmup_stream()
                warmup.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(warmup):
                    for _ in range(3):
                        self._run(bucket, paging, replay=False)
                torch.cuda.current_stream().wait_stream(warmup)
                # Warmup activations sit cached in the default pool; the graph pool can't use them.
                torch.cuda.empty_cache()

                graphs = []
                for segment in range(len(runner._layers) + 1):
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, pool=memory_pool):
                        self._segment(bucket, segment)
                    memory_pool = graph.pool()
                    graphs.append(graph)
                self.graphs[bucket] = graphs
        except torch.cuda.OutOfMemoryError:
            logger.exception("piecewise graph capture ran out of memory; running eagerly")
            self.graphs.clear()
            return

        if not self._self_check():
            logger.error("a piecewise graph disagrees with eager execution; running eagerly")
            self.graphs.clear()
        torch.cuda.empty_cache()

    def run_eager_dummy(self, n_tokens: int) -> None:
        """One eager iteration of `n_tokens` tokens on dummy input: the path an iteration
        takes without graphs, and the activation peak a graphed one never exceeds."""
        if self._buffers is None or self._buffers.hidden.shape[1] < n_tokens:
            self._buffers = self._alloc_buffers(n_tokens)
        self._run(n_tokens, self._fill_dummy(n_tokens), replay=False)

    def replay(self, plan: BatchPlan, seqs: dict[int, Sequence], bucket: int) -> IterationResults:
        runner, buffers = self._runner, self._buffers
        with nvtx_range("prepare_inputs"):
            token_ids, positions, offsets = runner._flat_rows(plan, seqs)
            rows = runner._paging_rows(plan, seqs, offsets)
            pad = bucket - len(token_ids)
            index_rows = [
                token_ids + [0] * pad,
                positions + [0] * pad,
                rows.dest_block_id + [runner._scratch_block_id] * pad,
                rows.dest_within + [0] * pad,
            ]
            buffers.index_rows[:, :bucket].copy_(
                pinned(index_rows, torch.int64), non_blocking=True
            )
            paging = rows.to_device(runner.device)
        with nvtx_range("forward"):
            self._run(bucket, paging, replay=True)
        return runner.emit_owed(buffers.hidden, plan, seqs, offsets)

    @torch.no_grad()
    def _run(self, bucket: int, paging: PagedBatch, replay: bool) -> None:
        """Segment 0, then for each layer: eager attention into `attn`, the next segment."""
        runner, buffers = self._runner, self._buffers

        def run_segment(segment: int) -> None:
            if replay:
                self.graphs[bucket][segment].replay()
            else:
                self._segment(bucket, segment)

        run_segment(0)
        for layer_idx in range(len(runner._layers)):
            attn = runner.attend_paged(buffers.q[:, :bucket], layer_idx, paging)
            buffers.attn[:, :bucket].copy_(attn)
            run_segment(layer_idx + 1)

    @torch.no_grad()
    def _segment(self, bucket: int, segment: int) -> None:
        runner, buffers = self._runner, self._buffers
        if segment == 0:
            hidden = runner.model.get_input_embeddings()(buffers.index_rows[0:1, :bucket])
            cos, sin = runner.model.model.rotary_emb(hidden, buffers.index_rows[1:2, :bucket])
            buffers.cos[:, :bucket].copy_(cos)
            buffers.sin[:, :bucket].copy_(sin)
            self._enter_layer(bucket, hidden, 0)
            return

        layer_idx = segment - 1
        layer = runner._layers[layer_idx]
        hidden = buffers.residual[:, :bucket] + runner.attention_output(
            buffers.attn[:, :bucket], layer_idx
        )
        hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
        if segment < len(runner._layers):
            self._enter_layer(bucket, hidden, segment)
        else:
            buffers.hidden[:, :bucket].copy_(runner.model.model.norm(hidden))

    def _enter_layer(self, bucket: int, hidden: torch.Tensor, layer_idx: int) -> None:
        """Saves `hidden` as the layer's residual, writes its K/V and stores its q."""
        runner, buffers = self._runner, self._buffers
        buffers.residual[:, :bucket].copy_(hidden)
        q = runner.write_kv_and_project_q(
            runner._layers[layer_idx].input_layernorm(hidden),
            (buffers.cos[:, :bucket], buffers.sin[:, :bucket]),
            layer_idx,
            buffers.index_rows[2, :bucket],
            buffers.index_rows[3, :bucket],
        )
        buffers.q[:, :bucket].copy_(q)

    def _alloc_buffers(self, max_tokens: int) -> SegmentBuffers:
        runner = self._runner
        d, dtype = runner.device, runner.model.dtype
        hidden_size = runner.model.config.hidden_size
        return SegmentBuffers(
            index_rows=torch.zeros(4, max_tokens, dtype=torch.int64, device=d),
            cos=torch.zeros(1, max_tokens, runner.head_dim, dtype=dtype, device=d),
            sin=torch.zeros(1, max_tokens, runner.head_dim, dtype=dtype, device=d),
            residual=torch.zeros(1, max_tokens, hidden_size, dtype=dtype, device=d),
            q=torch.zeros(runner.n_heads, max_tokens, runner.head_dim, dtype=dtype, device=d),
            attn=torch.zeros(runner.n_heads, max_tokens, runner.head_dim, dtype=dtype, device=d),
            hidden=torch.zeros(1, max_tokens, hidden_size, dtype=dtype, device=d),
        )

    def _fill_dummy(self, bucket: int) -> PagedBatch:
        """One entry spanning the whole bucket whose every block is the scratch block, so
        the full compute path runs and no block a sequence could own is touched. One entry,
        because the kernel launches entries x heads programs along a grid axis that CUDA
        caps at 65535."""
        runner, buffers = self._runner, self._buffers
        scratch, block_size, d = runner._scratch_block_id, runner._block_size, runner.device
        buffers.index_rows[:, :bucket].zero_()
        buffers.index_rows[1, :bucket] = torch.arange(bucket, device=d)
        buffers.index_rows[2, :bucket].fill_(scratch)
        buffers.index_rows[3, :bucket] = torch.arange(bucket, device=d) % block_size
        num_blocks = -(-bucket // block_size)
        return PagedBatch(
            block_table=torch.full((1, num_blocks), scratch, dtype=torch.int32, device=d),
            context_len=torch.tensor([bucket], dtype=torch.int32, device=d),
            query_offset=torch.zeros(1, dtype=torch.int32, device=d),
            q_start=torch.zeros(1, dtype=torch.int32, device=d),
            q_len=torch.tensor([bucket], dtype=torch.int32, device=d),
            max_q_len=bucket,
            dest_block_id=buffers.index_rows[2, :bucket],
            dest_within=buffers.index_rows[3, :bucket],
        )

    def _self_check(self) -> bool:
        """Replays each bucket against eager execution of the same segments on the same
        input. A bad capture can fault with a CUDA error that surfaces on a later, unrelated
        call, so this runs once at startup, before any request reaches a graph."""
        buffers = self._buffers
        for bucket in self.graphs:
            paging = self._fill_dummy(bucket)
            self._run(bucket, paging, replay=False)
            eager = buffers.hidden[:, :bucket].clone()
            self._run(bucket, paging, replay=True)
            torch.cuda.synchronize()
            if not torch.allclose(buffers.hidden[:, :bucket], eager, atol=1e-2, rtol=1e-2):
                return False
        return True
