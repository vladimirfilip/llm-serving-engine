"""CUDA graph capture and replay for the decode step.

Captures one CUDA graph per decode batch-size bucket, replacing forward_fused's
per-layer eager dispatch (~30 launches x n_layers) with a single replay. Only
pure-decode plans (every entry a 1-token decode step) ever use these; ModelRunner.forward
falls back to forward_fused for prefill/mixed plans or a batch larger than the largest
captured bucket.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .batch_plan import BatchPlan
from .paged_batch import PagedBatch
from .sampling import sample_token

if TYPE_CHECKING:
    import torch

    from .model_runner import ModelRunner
    from .sequence import Sequence

logger = logging.getLogger(__name__)

# Geometric spacing bounds padding waste to under 2x at any real batch size; chosen to
# cover MAX_CONCURRENT_SEQUENCES (scheduler.py) without committing to every size.
DECODE_GRAPH_BUCKETS = [1, 2, 4, 8, 16]


@dataclass(slots=True)
class DecodeGraphBuffers:
    """Fixed-address input/output tensors for one bucket size's captured graph.
    q_start is arange(bucket): a decode-only batch always packs exactly one token per
    entry, so entry i's flattened position is always i -- fixed for the buffer's life,
    never rewritten per iteration. Every other field is rewritten every iteration (see
    DecodeGraphRunner._fill_real) because the graph only replays whatever these
    addresses hold at replay time, not whatever they held at capture time.
    """

    input_ids: torch.Tensor  # (1, bucket)
    position_ids: torch.Tensor  # (1, bucket)
    block_table: torch.Tensor  # (bucket, num_blocks)
    context_len: torch.Tensor  # (bucket,)
    query_offset: torch.Tensor  # (bucket,)
    q_start: torch.Tensor  # (bucket,), constant
    q_len: torch.Tensor  # (bucket,), 0 for padding rows
    dest_block_id: torch.Tensor  # (bucket,)
    dest_within: torch.Tensor  # (bucket,)
    logits: torch.Tensor  # (bucket, vocab_size), captured-region output


@dataclass(slots=True)
class DecodeGraph:
    graph: torch.cuda.CUDAGraph
    buffers: DecodeGraphBuffers


class DecodeGraphRunner:
    """Owns every captured decode graph for one ModelRunner: buffer allocation,
    capture, the startup self-check, and per-iteration replay.

    `runner` supplies the actual compute (layers, attention_forward, ffn_forward,
    embeddings/lm_head) and the paged pool's layout (block_size, scratch_block_id) --
    this class owns only the graphs themselves and their fixed-address buffers.
    Behaves like a read-only `dict[int, DecodeGraph]` (bucket size -> graph) for
    callers that just need to check readiness or look up a captured bucket.
    """

    def __init__(self, runner: ModelRunner) -> None:
        self._runner = runner
        self.graphs: dict[int, DecodeGraph] = {}

    def __bool__(self) -> bool:
        return bool(self.graphs)

    def __iter__(self):
        return iter(self.graphs)

    def __getitem__(self, bucket: int) -> DecodeGraph:
        return self.graphs[bucket]

    def bucket_for(self, n: int) -> int | None:
        """Smallest captured bucket that covers a decode batch of size n, or None if
        every bucket is too small."""
        return min((b for b in self.graphs if b >= n), default=None)

    def _step_core(self, buffers: DecodeGraphBuffers) -> None:
        """Embedding through the batched lm_head, writing into buffers.logits -- the
        region capture() records, and also run eagerly (uncaptured) for warmup and the
        self-check, since it's the same call either way.

        No sampling here. sample_token branches in Python on each sequence's
        SamplingParams, and which branch runs for a given batch row changes every
        iteration as `running` membership changes -- a captured graph freezes which
        kernels ran, not just their inputs, so baking sampling in would silently replay
        whatever algorithm happened to occupy a row at capture time onto whatever
        different request occupies that row later. Sampling runs eagerly after replay
        instead, in replay().
        """
        import torch

        runner = self._runner
        hidden = runner.model.get_input_embeddings()(buffers.input_ids)
        position_embeddings = runner.model.model.rotary_emb(hidden, buffers.position_ids)
        paging = PagedBatch(
            block_table=buffers.block_table, context_len=buffers.context_len,
            query_offset=buffers.query_offset, q_start=buffers.q_start, q_len=buffers.q_len,
            max_q_len=1, dest_block_id=buffers.dest_block_id, dest_within=buffers.dest_within,
        )
        empty_plan, empty_seqs = BatchPlan(), {}  # dead params: attention_forward never
        # reads plan/seqs/offsets once paging is not None

        with torch.no_grad():
            for layer_idx, layer in enumerate(runner._layers):
                attn_out = runner.attention_forward(
                    layer.input_layernorm(hidden), empty_plan, empty_seqs, [], position_embeddings,
                    layer_idx, paging,
                )
                hidden = hidden + attn_out
                ffn_out = runner.ffn_forward(layer.post_attention_layernorm(hidden), layer_idx)
                hidden = hidden + ffn_out
            hidden = runner.model.model.norm(hidden)
            buffers.logits.copy_(runner.model.get_output_embeddings()(hidden[0]))

    def _alloc_buffers(self, bucket: int, num_blocks: int) -> DecodeGraphBuffers:
        import torch

        runner = self._runner
        cfg = runner.model.config
        dtype = next(runner.model.parameters()).dtype
        d = runner.device
        return DecodeGraphBuffers(
            input_ids=torch.zeros(1, bucket, dtype=torch.int64, device=d),
            position_ids=torch.zeros(1, bucket, dtype=torch.int64, device=d),
            block_table=torch.zeros(bucket, num_blocks, dtype=torch.int32, device=d),
            context_len=torch.zeros(bucket, dtype=torch.int32, device=d),
            query_offset=torch.zeros(bucket, dtype=torch.int32, device=d),
            q_start=torch.arange(bucket, dtype=torch.int32, device=d),
            q_len=torch.zeros(bucket, dtype=torch.int32, device=d),
            dest_block_id=torch.zeros(bucket, dtype=torch.int64, device=d),
            dest_within=torch.zeros(bucket, dtype=torch.int64, device=d),
            logits=torch.zeros(bucket, cfg.vocab_size, dtype=dtype, device=d),
        )

    def _fill_dummy(self, buffers: DecodeGraphBuffers) -> None:
        """Shape- and index-valid placeholder data for warmup and the self-check: every
        row marked real (q_len=1) so the full compute path runs, not just the padding
        early-return, and every row's K/V write points at the scratch block so nothing
        touches a block a real sequence could ever own."""
        import torch

        scratch = self._runner._scratch_block_id
        bucket = buffers.q_len.shape[0]
        d = self._runner.device
        buffers.input_ids.zero_()
        buffers.position_ids.zero_()
        buffers.block_table.fill_(scratch)
        buffers.context_len.fill_(1)
        buffers.query_offset.zero_()
        buffers.q_len.fill_(1)
        buffers.dest_block_id.copy_(torch.full((bucket,), scratch, dtype=torch.int64, device=d))
        buffers.dest_within.zero_()

    def _fill_real(
        self, buffers: DecodeGraphBuffers, plan: BatchPlan, seqs: dict[int, Sequence]
    ) -> None:
        """Fills every row for this iteration's real decode plan, padding the rest with
        q_len=0 (skips the kernel's read for that row entirely -- see paged_attention_2's
        early return) and the reserved scratch block (so the pool write, which has no
        mask at all, can never land on a block a live sequence owns).

        Every field is rewritten for every row, every iteration, real or padding: which
        rows are "padding" changes call to call as sequences finish and new ones are
        admitted, and a stale q_len or dest_block_id left over from a row's previous
        occupant would defeat the guards above just as surely as never setting them.
        """
        import torch

        runner = self._runner
        bucket = buffers.q_len.shape[0]
        block_size = runner._block_size
        scratch = runner._scratch_block_id
        max_blocks = buffers.block_table.shape[1]

        input_ids = [0] * bucket
        position_ids = [0] * bucket
        context_len = [0] * bucket
        query_offset = [0] * bucket
        q_len = [0] * bucket
        dest_block_id = [scratch] * bucket
        dest_within = [0] * bucket
        block_table_rows = [[scratch] + [0] * (max_blocks - 1) for _ in range(bucket)]

        for i, entry in enumerate(plan):
            seq = seqs[entry.seq_id]
            pos = seq.block_table.num_tokens - 1  # position of the new token this call writes
            physical = seq.block_table.physical_blocks
            input_ids[i] = seq.generated_tokens[-1]
            position_ids[i] = pos
            context_len[i] = pos + 1
            query_offset[i] = pos
            q_len[i] = 1
            dest_block_id[i] = physical[pos // block_size]
            dest_within[i] = pos % block_size
            block_table_rows[i][: len(physical)] = physical

        d = runner.device
        buffers.input_ids.copy_(torch.tensor([input_ids], dtype=torch.int64, device=d))
        buffers.position_ids.copy_(torch.tensor([position_ids], dtype=torch.int64, device=d))
        buffers.block_table.copy_(torch.tensor(block_table_rows, dtype=torch.int32, device=d))
        buffers.context_len.copy_(torch.tensor(context_len, dtype=torch.int32, device=d))
        buffers.query_offset.copy_(torch.tensor(query_offset, dtype=torch.int32, device=d))
        buffers.q_len.copy_(torch.tensor(q_len, dtype=torch.int32, device=d))
        buffers.dest_block_id.copy_(torch.tensor(dest_block_id, dtype=torch.int64, device=d))
        buffers.dest_within.copy_(torch.tensor(dest_within, dtype=torch.int64, device=d))

    def capture(self, bucket_sizes: list[int] = DECODE_GRAPH_BUCKETS) -> None:
        """No-ops if the runner's KV pool hasn't been allocated -- graphs need the pool
        to write into. Capture failure (e.g. insufficient GPU memory on top of the KV
        pool) and self-check failure both leave self.graphs empty rather than raising: a
        server that can't capture graphs should still start and serve on forward_fused,
        not fail to boot over an opt-in fast path.
        """
        import torch

        runner = self._runner
        if runner._k_pool is None:
            return

        num_blocks = runner._k_pool[0].shape[0] - 1  # last row is the scratch block
        pool_handle = None
        try:
            for bucket in sorted(bucket_sizes):
                buffers = self._alloc_buffers(bucket, num_blocks)
                self._fill_dummy(buffers)

                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        self._step_core(buffers)
                torch.cuda.current_stream().wait_stream(stream)
                torch.cuda.synchronize()

                graph = torch.cuda.CUDAGraph()
                capture_kwargs = {"pool": pool_handle} if pool_handle is not None else {}
                with torch.cuda.graph(graph, **capture_kwargs):
                    self._step_core(buffers)
                pool_handle = graph.pool()

                self.graphs[bucket] = DecodeGraph(graph=graph, buffers=buffers)
        except Exception:
            logger.exception(
                "CUDA graph capture failed; falling back to forward_fused for every iteration"
            )
            self.graphs.clear()
            return

        if not self._self_check():
            logger.error(
                "a captured decode graph didn't match forward_fused's output on synthetic "
                "data; falling back to forward_fused for every iteration"
            )
            self.graphs.clear()

    def _self_check(self) -> bool:
        """Runs _step_core eagerly and via each captured graph on identical synthetic
        input and compares the two. A bad capture can produce an illegal memory access
        whose CUDA error surfaces on a later, unrelated call rather than on the replay()
        that actually caused it -- this is what keeps a broken capture from ever reaching
        a real request instead of catching it here, once, at startup."""
        import torch

        for dg in self.graphs.values():
            self._fill_dummy(dg.buffers)
            self._step_core(dg.buffers)
            eager_logits = dg.buffers.logits.clone()

            dg.graph.replay()
            torch.cuda.synchronize()

            if not torch.allclose(dg.buffers.logits, eager_logits, atol=1e-2, rtol=1e-2):
                return False
        return True

    def replay(
        self, plan: BatchPlan, seqs: dict[int, Sequence], bucket: int
    ) -> list[tuple[int, int, bool]]:
        """Pure-decode fast path: fills `bucket`'s static buffers with this iteration's
        real values, replays its captured graph, then samples eagerly from the resulting
        logits (see _step_core for why sampling isn't captured)."""
        dg = self.graphs[bucket]
        self._fill_real(dg.buffers, plan, seqs)
        dg.graph.replay()

        import torch

        tokens = torch.stack([
            sample_token(dg.buffers.logits[i], seqs[entry.seq_id].sampling_params)
            for i, entry in enumerate(plan)
        ]).tolist()

        results = []
        for entry, token_id in zip(plan, tokens, strict=True):
            seq = seqs[entry.seq_id]
            finished = token_id in self._runner.eos_token_ids or (
                len(seq.generated_tokens) + 1 >= seq.sampling_params.max_tokens
            )
            results.append((seq.seq_id, token_id, finished))
        return results
