"""Model loading and the GPU worker's per-iteration forward pass.

Two forward paths, chosen by config.use_custom_kernels:
- forward: one HF call per BatchPlan entry, each Sequence keyed into its own
  HuggingFace past_key_values cache.
- forward_fused: flattens the BatchPlan into one [1, total_tokens] tensor and runs
  the decoder layer by layer. K/V storage depends on the allocator: a paged pool
  (allocate_kv_pool) lets one Triton launch per layer cover the whole batch; a
  contiguous allocator falls back to one disjoint per-sequence buffer and one
  launch per entry.

A third path, forward_graphed, replaces forward_fused's ~30-launch-per-layer eager
dispatch with a single CUDA graph replay, for pure-decode iterations only. That
machinery (capture, self-check, replay) lives in decode_graph.DecodeGraphRunner;
this module only fills real per-iteration data into it and dispatches to it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .batch_plan import BatchEntry, BatchPlan
from .config import ModelConfig
from .decode_graph import DECODE_GRAPH_BUCKETS, DecodeGraphRunner
from .paged_batch import PagedBatch
from .sampling import SamplingParams, sample, sample_token
from .sequence import Sequence

if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedModel


class ModelRunner:
    def __init__(self, config: ModelConfig):
        import torch
        from transformers import AutoModelForCausalLM

        if config.quantize == "int8":
            raise NotImplementedError("int8 weight-only quantization on the hot path")

        self.config = config
        self._past_key_values: dict[int, Any] = {}
        self._kv_cache: dict[int, dict[int, tuple[Any, Any, int]]] = {}  # layer_idx -> seq_id -> (K, V, filled)
        self._k_pool: list[torch.Tensor] | None = None  # layer_idx -> (num_blocks, block_size, n_kv_heads, D)
        self._v_pool: list[torch.Tensor] | None = None
        self._block_size: int | None = None
        self._scratch_block_id: int | None = None  # padding rows' pool write target; see allocate_kv_pool
        self._decode_graphs = DecodeGraphRunner(self)

        self.device = config.device if torch.cuda.is_available() else "cpu"
        dtype = getattr(torch, config.dtype)  # an unknown dtype name must fail loudly
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path, dtype=dtype
        ).to(self.device)
        self.model.eval()

        if config.use_custom_kernels:
            self.wire_custom_kernels()

    def allocate_kv_pool(self, num_blocks: int, block_size: int) -> None:
        """One (num_blocks + 1, block_size, n_kv_heads, head_dim) K/V buffer per layer,
        shared across every sequence and addressed by the same physical block ids
        BlockAllocator hands out. Called once, after num_blocks is sized off free GPU
        memory, before the first forward().

        The pool has one row more than BlockAllocator ever hands out: block id
        num_blocks itself, reserved as a scratch write target for a graphed decode
        batch's padding rows. Those rows' K/V write is unconditional -- it happens in
        Python before the attention kernel launches, with no mask -- so without a
        dedicated target it would scatter into whatever block a live sequence owns.
        """
        import torch

        cfg = self.model.config
        n_heads = cfg.num_attention_heads
        n_kv_heads = cfg.num_key_value_heads
        head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // n_heads
        dtype = next(self.model.parameters()).dtype

        self._block_size = block_size
        self._scratch_block_id = num_blocks
        shape = (num_blocks + 1, block_size, n_kv_heads, head_dim)
        self._k_pool = [torch.empty(shape, device=self.device, dtype=dtype) for _ in self._layers]
        self._v_pool = [torch.empty(shape, device=self.device, dtype=dtype) for _ in self._layers]

    def wire_custom_kernels(self) -> None:
        """forward() already dispatches to forward_fused per call, so nothing here needs
        monkeypatching — this only checks forward_fused's preconditions once at load time,
        so a bad config fails now instead of on the first request it serves."""
        import torch

        if not self.device.startswith("cuda"):
            raise RuntimeError(f"custom kernels need a CUDA device, got {self.device!r}")
        if not hasattr(self.model, "model") or not hasattr(self.model.model, "layers"):
            raise RuntimeError(
                f"custom kernels assume a Llama-family decoder (model.model.layers); "
                f"{type(self.model).__name__} has no such attribute"
            )
        dtype = next(self.model.parameters()).dtype
        if dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise RuntimeError(f"custom kernels don't support dtype {dtype}")

        cfg = self.model.config
        head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        if dtype in (torch.float16, torch.bfloat16) and head_dim < 16:
            # fp16/bf16 route Q@K^T through a tensor-core tl.dot, which needs its
            # contraction dimension >= 16; fp32 uses the "ieee" (non-tensor-core) path
            # and isn't bound by this.
            raise RuntimeError(
                f"custom kernels need head_dim >= 16 in {dtype}, got {head_dim} "
                f"(hidden_size={cfg.hidden_size}, num_attention_heads={cfg.num_attention_heads})"
            )

    def build_token_id_tensors(self, entry: BatchEntry, seq: Sequence) -> torch.Tensor:
        """input_ids for one BatchPlan entry: the new prompt-token slice for a prefill
        chunk (scheduler_step already advanced prefill_progress past it), or the
        just-generated token for a decode step -- generated_tokens[-1] is only valid
        because handle_iteration_results appends before this runs."""
        import torch

        if entry.is_prefill_chunk:
            token_ids = seq.prompt_tokens[seq.prefill_progress - entry.n_tokens : seq.prefill_progress]
        else:
            token_ids = [seq.generated_tokens[-1]]
        return torch.tensor([token_ids], device=self.device)

    def forward(self, plan: BatchPlan, seqs: dict[int, Sequence]) -> list[tuple[int, int, bool]]:
        """One GPU-worker iteration against per-sequence KV caches. A prefill entry that
        doesn't finish the prompt (seq.status still "PREFILLING") extends the cache but
        samples nothing -- no token to report until the prompt is fully in cache."""
        import torch

        if self.config.use_custom_kernels:
            if self._decode_graphs and all(not entry.is_prefill_chunk for entry in plan):
                bucket = self._decode_graphs.bucket_for(len(plan))
                if bucket is not None:
                    return self.forward_graphed(plan, seqs, bucket)
            return self.forward_fused(plan, seqs)

        results = []
        for entry in plan:
            seq = seqs[entry.seq_id]
            input_ids = self.build_token_id_tensors(entry, seq)
            with torch.no_grad():
                out = self.model(
                    input_ids, past_key_values=self._past_key_values.get(seq.seq_id), use_cache=True
                )
            self._past_key_values[seq.seq_id] = out.past_key_values
            if entry.is_prefill_chunk and seq.status == "PREFILLING":
                continue
            token_id = self._sample(out.logits[0, -1], seq.sampling_params)
            finished = token_id in self.eos_token_ids or (
                len(seq.generated_tokens) + 1 >= seq.sampling_params.max_tokens
            )
            results.append((seq.seq_id, token_id, finished))
        return results

    @property
    def _layers(self) -> torch.nn.ModuleList:
        """Assumes a Llama-family decoder: self_attn/mlp/input_layernorm/
        post_attention_layernorm per layer."""
        return self.model.model.layers

    def _flatten_plan(
        self, plan: BatchPlan, seqs: dict[int, Sequence]
    ) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int]]]:
        """Concatenates every entry's tokens into one [1, total_tokens] sequence -- the
        layout attention_forward/ffn_forward operate on. offsets[i] is entry i's (start,
        end) span in that tensor, in plan order."""
        import torch

        input_ids, position_ids, offsets = [], [], []
        cursor = 0
        for entry in plan:
            seq = seqs[entry.seq_id]
            input_ids.append(self.build_token_id_tensors(entry, seq))
            start_pos = (
                seq.prefill_progress - entry.n_tokens if entry.is_prefill_chunk else seq.num_tokens - 1
            )
            position_ids.append(torch.arange(start_pos, start_pos + entry.n_tokens))
            offsets.append((cursor, cursor + entry.n_tokens))
            cursor += entry.n_tokens

        return (
            torch.cat(input_ids, dim=1).to(self.device),
            torch.cat(position_ids).unsqueeze(0).to(self.device),
            offsets,
        )

    def _prepare_paging(
        self, plan: BatchPlan, seqs: dict[int, Sequence], offsets: list[tuple[int, int]]
    ) -> PagedBatch:
        """Builds PagedBatch once per iteration, not once per layer. Reads each Sequence's
        BlockTable, which scheduler_step already updated before forward() runs -- so
        table.num_tokens is already this call's post-write total."""
        import torch

        block_size = self._block_size
        tables = [seqs[e.seq_id].block_table for e in plan]
        max_blocks = max(len(t.physical_blocks) for t in tables)

        block_table_rows: list[list[int]] = []
        context_len: list[int] = []
        query_offset: list[int] = []
        q_start: list[int] = []
        q_len: list[int] = []
        dest_block_id: list[int] = []
        dest_within: list[int] = []
        max_q_len = 0

        for table, (start, end) in zip(tables, offsets, strict=True):
            n_new = end - start
            max_q_len = max(max_q_len, n_new)
            cl = table.num_tokens
            qo = cl - n_new
            physical = table.physical_blocks
            block_table_rows.append(physical + [0] * (max_blocks - len(physical)))
            context_len.append(cl)
            query_offset.append(qo)
            q_start.append(start)
            q_len.append(n_new)
            dest_block_id.extend(physical[p // block_size] for p in range(qo, cl))
            dest_within.extend(p % block_size for p in range(qo, cl))

        return PagedBatch(
            block_table=torch.tensor(block_table_rows, dtype=torch.int32, device=self.device),
            context_len=torch.tensor(context_len, dtype=torch.int32, device=self.device),
            query_offset=torch.tensor(query_offset, dtype=torch.int32, device=self.device),
            q_start=torch.tensor(q_start, dtype=torch.int32, device=self.device),
            q_len=torch.tensor(q_len, dtype=torch.int32, device=self.device),
            max_q_len=max_q_len,
            dest_block_id=torch.tensor(dest_block_id, dtype=torch.int64, device=self.device),
            dest_within=torch.tensor(dest_within, dtype=torch.int64, device=self.device),
        )

    def _paged_attention_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        paging: PagedBatch,
        layer_idx: int,
        total_tokens: int,
    ) -> torch.Tensor:
        """One Triton launch for the whole BatchPlan's attention at this layer: writes
        this call's new K/V into the pool at PagedBatch's resolved slots, then gathers
        through each entry's block table (per-row addressing: paged_attention_2)."""
        from .kernels.flash_attention import paged_attention_forward

        k_pool, v_pool = self._k_pool[layer_idx], self._v_pool[layer_idx]
        k_pool[paging.dest_block_id, paging.dest_within] = k[0].transpose(0, 1)
        v_pool[paging.dest_block_id, paging.dest_within] = v[0].transpose(0, 1)

        out = paged_attention_forward(
            q[0], k_pool, v_pool, paging.block_table, paging.context_len,
            paging.query_offset, paging.q_start, paging.q_len, self._block_size, paging.max_q_len,
        )  # (n_heads, total_tokens, head_dim)
        return out.transpose(0, 1).unsqueeze(0).reshape(1, total_tokens, -1)

    def attention_forward(
        self,
        normed: torch.Tensor,
        plan: BatchPlan,
        seqs: dict[int, Sequence],
        offsets: list[tuple[int, int]],
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        layer_idx: int,
        paging: PagedBatch | None = None,
    ) -> torch.Tensor:
        """FlashAttention (kernels/flash_attention.py). With a paged pool (`paging` is
        not None), one launch covers every plan entry against the shared block-addressed
        pool. Without one, each entry gets its own launch against a disjoint per-sequence
        buffer, since each entry's queries must only see its own sequence's cache."""
        import torch
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

        from .kernels.flash_attention import flash_attention_forward

        attn = self._layers[layer_idx].self_attn
        cfg = self.model.config
        n_heads, n_kv_heads = cfg.num_attention_heads, cfg.num_key_value_heads
        head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // n_heads
        n_groups = n_heads // n_kv_heads
        total_tokens = normed.shape[1]

        q = attn.q_proj(normed).view(1, total_tokens, n_heads, head_dim).transpose(1, 2)
        k = attn.k_proj(normed).view(1, total_tokens, n_kv_heads, head_dim).transpose(1, 2)
        v = attn.v_proj(normed).view(1, total_tokens, n_kv_heads, head_dim).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, *position_embeddings)

        if paging is not None:
            out = self._paged_attention_forward(q, k, v, paging, layer_idx, total_tokens)
            return attn.o_proj(out)

        cache = self._kv_cache.setdefault(layer_idx, {})
        out = torch.empty(1, total_tokens, n_heads, head_dim, device=normed.device, dtype=normed.dtype)

        for entry, (start, end) in zip(plan, offsets, strict=True):
            seq_q = q[:, :, start:end]
            n_new = end - start

            buf = cache.get(entry.seq_id)
            if buf is None:
                # Sized to prompt + max_tokens (worst-case total, same bound
                # ContiguousAllocator reserves) and filled by index below, not torch.cat --
                # cat would recopy the whole cached span every decode step.
                capacity = len(seqs[entry.seq_id].prompt_tokens) + seqs[entry.seq_id].sampling_params.max_tokens
                k_buf = torch.empty(1, n_kv_heads, capacity, head_dim, device=k.device, dtype=k.dtype)
                v_buf = torch.empty_like(k_buf)
                filled = 0
            else:
                k_buf, v_buf, filled = buf

            k_buf[:, :, filled : filled + n_new] = k[:, :, start:end]
            v_buf[:, :, filled : filled + n_new] = v[:, :, start:end]
            filled += n_new
            cache[entry.seq_id] = (k_buf, v_buf, filled)  # disjoint per-sequence buffer, not a shared pool

            # query_offset + row is a query's true position; the kernel's causal mask
            # compares that against each key position, so a decode step's one new token
            # still attends to its whole cache.
            query_offset = filled - n_new
            seq_k = k_buf[:, :, :filled].repeat_interleave(n_groups, dim=1)  # GQA: kv heads -> q heads
            seq_v = v_buf[:, :, :filled].repeat_interleave(n_groups, dim=1)

            entry_out, _ = flash_attention_forward(
                seq_q, seq_k, seq_v, is_causal=True, query_offset=query_offset
            )
            out[:, start:end] = entry_out.transpose(1, 2)

        return attn.o_proj(out.reshape(1, total_tokens, n_heads * head_dim))

    def ffn_forward(self, normed: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """FFN tokens are independent, so unlike attention_forward this is one call over
        the whole flattened tensor."""
        return self._layers[layer_idx].mlp(normed)

    def forward_fused(self, plan: BatchPlan, seqs: dict[int, Sequence]) -> list[tuple[int, int, bool]]:
        """Fused GPU-worker iteration: embeds the BatchPlan once, runs
        attention_forward/ffn_forward per layer, then samples only rows that owe a token
        (decode entries, and prefill entries whose chunk just completed the prompt)."""
        import torch

        input_ids, position_ids, offsets = self._flatten_plan(plan, seqs)
        hidden = self.model.get_input_embeddings()(input_ids)
        # cos/sin depend only on position_ids and head_dim, not layer_idx — computed once
        # here rather than inside attention_forward's per-layer loop.
        position_embeddings = self.model.model.rotary_emb(hidden, position_ids)
        paging = self._prepare_paging(plan, seqs, offsets) if self._k_pool is not None else None

        with torch.no_grad():
            for layer_idx, layer in enumerate(self._layers):
                attn_out = self.attention_forward(
                    layer.input_layernorm(hidden), plan, seqs, offsets, position_embeddings,
                    layer_idx, paging,
                )
                hidden = hidden + attn_out
                ffn_out = self.ffn_forward(layer.post_attention_layernorm(hidden), layer_idx)
                hidden = hidden + ffn_out
            hidden = self.model.model.norm(hidden)

            eligible = [
                (entry, end - 1)
                for entry, (_start, end) in zip(plan, offsets, strict=True)
                if not (entry.is_prefill_chunk and seqs[entry.seq_id].status == "PREFILLING")
            ]
            results = []
            if eligible:
                # One lm_head call over every entry owing a token
                positions = torch.tensor([p for _, p in eligible], device=hidden.device)
                logits = self.model.get_output_embeddings()(hidden[0, positions])
                tokens = torch.stack([
                    self._sample_token(logits[i], seqs[entry.seq_id].sampling_params)
                    for i, (entry, _pos) in enumerate(eligible)
                ]).tolist()
                for (entry, _pos), token_id in zip(eligible, tokens, strict=True):
                    seq = seqs[entry.seq_id]
                    finished = token_id in self.eos_token_ids or (
                        len(seq.generated_tokens) + 1 >= seq.sampling_params.max_tokens
                    )
                    results.append((seq.seq_id, token_id, finished))
        return results

    def capture_decode_graphs(self, bucket_sizes: list[int] = DECODE_GRAPH_BUCKETS) -> None:
        self._decode_graphs.capture(bucket_sizes)

    def forward_graphed(
        self, plan: BatchPlan, seqs: dict[int, Sequence], bucket: int
    ) -> list[tuple[int, int, bool]]:
        return self._decode_graphs.replay(plan, seqs, bucket)

    def free(self, seq_id: int) -> None:
        """Drops a finished/cancelled sequence's KV cache; call alongside allocator.free
        from result handling. The paged pool needs nothing here -- its blocks are
        addressed by id, not seq_id, and allocator.free already freed them."""
        self._past_key_values.pop(seq_id, None)
        for layer_cache in self._kv_cache.values():
            layer_cache.pop(seq_id, None)

    def _sample(self, logits: torch.Tensor, params: SamplingParams) -> int:
        return sample(logits, params)

    def _sample_token(self, logits: torch.Tensor, params: SamplingParams) -> torch.Tensor:
        return sample_token(logits, params)

    @property
    def eos_token_ids(self) -> frozenset[int]:
        # Instruct-tuned models (Llama-3's <|eot_id|> alongside <|end_of_text|>) list more
        # than one stop token; matching only the first silently never finishes on the rest.
        eos = self.model.generation_config.eos_token_id
        return frozenset(eos) if isinstance(eos, list) else frozenset({eos})
