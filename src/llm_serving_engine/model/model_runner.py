"""Model loading and the GPU worker's per-iteration forward pass.

`forward` takes one of three paths:
- Without custom kernels, one HF call per plan entry, against that sequence's own
  `past_key_values`.
- `forward_fused` flattens the plan into one (1, total_tokens) batch and runs the decoder
  layer by layer on Triton attention. With a paged pool, one attention launch per layer
  covers every entry; with the contiguous allocator, each entry attends over its own buffer.
- A pure-decode plan that fits a captured bucket replays that bucket's CUDA graph.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any, Callable

import torch
from transformers import AutoModelForCausalLM
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from ..config import ModelConfig
from ..kernels.flash_attention import flash_attention_forward, paged_attention_forward
from ..scheduling.batch_plan import BatchEntry, BatchPlan
from ..scheduling.sequence import Sequence
from .decode_graph import DecodeGraphRunner
from .paged_batch import PagedBatch, pinned
from .sampling import sample_token

if TYPE_CHECKING:
    from transformers import PreTrainedModel

# (seq_id, token, finished) for each sequence that owes a token this iteration.
IterationResults = list[tuple[int, int, bool]]
# One layer's attention: (normed hidden, rotary (cos, sin), layer_idx) -> attention output.
Attend = Callable[[torch.Tensor, tuple[torch.Tensor, torch.Tensor], int], torch.Tensor]


class ModelRunner:
    def __init__(self, config: ModelConfig):
        if config.quantize == "int8":
            raise NotImplementedError("int8 weight-only quantization")

        self.config = config
        self.device = config.device if torch.cuda.is_available() else "cpu"
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path, dtype=getattr(torch, config.dtype)
        ).to(self.device)
        self.model.eval()

        cfg = self.model.config
        self.n_heads = cfg.num_attention_heads
        self.n_kv_heads = getattr(cfg, "num_key_value_heads", self.n_heads)
        self.head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // self.n_heads
        self.eos_token_ids = eos_token_ids(self.model.generation_config.eos_token_id)

        self._past_key_values: dict[int, Any] = {}
        # layer_idx -> seq_id -> (K, V, filled); K and V are (1, n_kv_heads, capacity, head_dim).
        self._kv_cache: dict[int, dict[int, tuple[torch.Tensor, torch.Tensor, int]]] = {}
        # Per layer, (num_blocks + 1, block_size, n_kv_heads, head_dim).
        self._k_pool: list[torch.Tensor] | None = None
        self._v_pool: list[torch.Tensor] | None = None
        self._block_size: int | None = None
        self._scratch_block_id: int | None = None
        self._decode_graphs = DecodeGraphRunner(self)

        if config.use_custom_kernels:
            self.check_custom_kernel_support()

    def check_custom_kernel_support(self) -> None:
        """Fails at load time on a device, model or dtype the Triton path can't run."""
        if not self.device.startswith("cuda"):
            raise RuntimeError(f"custom kernels need a CUDA device, got {self.device!r}")
        if not hasattr(self.model, "model") or not hasattr(self.model.model, "layers"):
            raise RuntimeError(
                f"custom kernels assume a Llama-family decoder (model.model.layers); "
                f"{type(self.model).__name__} has no such attribute"
            )
        dtype = self.model.dtype
        if dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise RuntimeError(f"custom kernels don't support dtype {dtype}")
        if dtype != torch.float32 and self.head_dim < 16:
            # fp16/bf16 compute Q@K^T with a tensor-core tl.dot, whose contraction
            # dimension must be at least 16.
            raise RuntimeError(
                f"custom kernels need head_dim >= 16 in {dtype}, got {self.head_dim}"
            )

    def allocate_kv_pool(self, num_blocks: int, block_size: int) -> None:
        """One K and one V pool per layer, addressed by the block ids BlockAllocator hands
        out. Row `num_blocks` is never handed out: it is the write target for a graphed
        decode's padding rows, whose K/V write is unmasked."""
        shape = (num_blocks + 1, block_size, self.n_kv_heads, self.head_dim)
        dtype = self.model.dtype
        self._block_size = block_size
        self._scratch_block_id = num_blocks
        self._k_pool = [torch.empty(shape, device=self.device, dtype=dtype) for _ in self._layers]
        self._v_pool = [torch.empty(shape, device=self.device, dtype=dtype) for _ in self._layers]

    def capture_decode_graphs(self, bucket_sizes: list[int]) -> None:
        self._decode_graphs.capture(bucket_sizes)

    def forward(self, plan: BatchPlan, seqs: dict[int, Sequence]) -> IterationResults:
        """One GPU-worker iteration. A prefill chunk that leaves its sequence PREFILLING
        extends the cache and samples nothing."""
        if not self.config.use_custom_kernels:
            return self._forward_hf(plan, seqs)
        if all(not entry.is_prefill_chunk for entry in plan):
            bucket = self._decode_graphs.bucket_for(len(plan))
            if bucket is not None:
                return self._decode_graphs.replay(plan, seqs, bucket)
        return self.forward_fused(plan, seqs)

    @torch.no_grad()
    def _forward_hf(self, plan: BatchPlan, seqs: dict[int, Sequence]) -> IterationResults:
        owed: list[Sequence] = []
        last_logits: list[torch.Tensor] = []
        for entry in plan:
            seq = seqs[entry.seq_id]
            input_ids = torch.tensor([_entry_token_ids(entry, seq)], device=self.device)
            out = self.model(
                input_ids, past_key_values=self._past_key_values.get(seq.seq_id), use_cache=True
            )
            self._past_key_values[seq.seq_id] = out.past_key_values
            if _owes_token(entry, seq):
                owed.append(seq)
                last_logits.append(out.logits[0, -1])
        if not owed:
            return []
        return self.emit_tokens(torch.stack(last_logits), owed)

    @torch.no_grad()
    def forward_fused(self, plan: BatchPlan, seqs: dict[int, Sequence]) -> IterationResults:
        input_ids, position_ids, offsets = self._flatten_plan(plan, seqs)
        if self._k_pool is not None:
            paging = self._prepare_paging(plan, seqs, offsets)
            attend = partial(self.paged_attention, paging=paging)
        else:
            attend = partial(self.contiguous_attention, plan=plan, seqs=seqs, offsets=offsets)
        hidden = self.decoder_hidden(input_ids, position_ids, attend)

        owed = [
            (seqs[entry.seq_id], end - 1)
            for entry, (_start, end) in zip(plan, offsets, strict=True)
            if _owes_token(entry, seqs[entry.seq_id])
        ]
        if not owed:
            return []
        last_rows = torch.tensor([row for _seq, row in owed], device=self.device)
        logits = self.model.get_output_embeddings()(hidden[0, last_rows])
        return self.emit_tokens(logits, [seq for seq, _row in owed])

    def decoder_hidden(
        self, input_ids: torch.Tensor, position_ids: torch.Tensor, attend: Attend
    ) -> torch.Tensor:
        """Embedding through the final norm: (1, total_tokens) ids -> (1, total_tokens,
        hidden_size). The MLP runs over the whole flattened batch; only attention needs
        per-entry structure, which `attend` supplies."""
        hidden = self.model.get_input_embeddings()(input_ids)
        position_embeddings = self.model.model.rotary_emb(hidden, position_ids)
        for layer_idx, layer in enumerate(self._layers):
            hidden = hidden + attend(layer.input_layernorm(hidden), position_embeddings, layer_idx)
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
        return self.model.model.norm(hidden)

    def emit_tokens(self, logits: torch.Tensor, owed: list[Sequence]) -> IterationResults:
        """Samples row i of `logits` for owed[i], with one host sync for the whole batch."""
        tokens = torch.stack(
            [sample_token(logits[i], seq.sampling_params) for i, seq in enumerate(owed)]
        ).tolist()
        return [
            (
                seq.seq_id,
                token,
                token in self.eos_token_ids
                or len(seq.generated_tokens) + 1 >= seq.sampling_params.max_tokens,
            )
            for seq, token in zip(owed, tokens, strict=True)
        ]

    def _flatten_plan(
        self, plan: BatchPlan, seqs: dict[int, Sequence]
    ) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int]]]:
        """input_ids and position_ids, each (1, total_tokens): every entry's tokens in plan
        order. offsets[i] is entry i's [start, end) span."""
        token_ids: list[int] = []
        positions: list[int] = []
        offsets: list[tuple[int, int]] = []
        for entry in plan:
            seq = seqs[entry.seq_id]
            if entry.is_prefill_chunk:
                start_pos = seq.prefill_progress - entry.n_tokens
            else:
                start_pos = seq.num_tokens - 1
            offsets.append((len(token_ids), len(token_ids) + entry.n_tokens))
            token_ids.extend(_entry_token_ids(entry, seq))
            positions.extend(range(start_pos, start_pos + entry.n_tokens))
        ids = pinned([token_ids, positions], torch.int64).to(self.device, non_blocking=True)
        return ids[0:1], ids[1:2], offsets

    def _prepare_paging(
        self, plan: BatchPlan, seqs: dict[int, Sequence], offsets: list[tuple[int, int]]
    ) -> PagedBatch:
        """Reads each BlockTable after scheduler_step grew it, so num_tokens already
        counts this call's tokens."""
        block_size = self._block_size
        tables = [seqs[entry.seq_id].block_table for entry in plan]
        max_blocks = max(len(table.physical_blocks) for table in tables)

        block_table: list[list[int]] = []
        context_len: list[int] = []
        query_offset: list[int] = []
        q_start: list[int] = []
        q_len: list[int] = []
        dest_block_id: list[int] = []
        dest_within: list[int] = []
        for table, (start, end) in zip(tables, offsets, strict=True):
            physical = table.physical_blocks
            cached = table.num_tokens - (end - start)
            block_table.append(physical + [0] * (max_blocks - len(physical)))
            context_len.append(table.num_tokens)
            query_offset.append(cached)
            q_start.append(start)
            q_len.append(end - start)
            for pos in range(cached, table.num_tokens):
                dest_block_id.append(physical[pos // block_size])
                dest_within.append(pos % block_size)

        d = self.device
        per_entry = pinned([context_len, query_offset, q_start, q_len], torch.int32)
        per_entry = per_entry.to(d, non_blocking=True)
        dest = pinned([dest_block_id, dest_within], torch.int64).to(d, non_blocking=True)
        return PagedBatch(
            block_table=pinned(block_table, torch.int32).to(d, non_blocking=True),
            context_len=per_entry[0],
            query_offset=per_entry[1],
            q_start=per_entry[2],
            q_len=per_entry[3],
            max_q_len=max(q_len),
            dest_block_id=dest[0],
            dest_within=dest[1],
        )

    def _project_qkv(
        self,
        normed: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """q (1, n_heads, total_tokens, head_dim); k, v (1, n_kv_heads, total_tokens,
        head_dim); q and k rotated."""
        attn = self._layers[layer_idx].self_attn
        total_tokens = normed.shape[1]
        q_shape = (1, total_tokens, self.n_heads, self.head_dim)
        kv_shape = (1, total_tokens, self.n_kv_heads, self.head_dim)
        q = attn.q_proj(normed).view(q_shape).transpose(1, 2)
        k = attn.k_proj(normed).view(kv_shape).transpose(1, 2)
        v = attn.v_proj(normed).view(kv_shape).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
        return q, k, v

    def paged_attention(
        self,
        normed: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        layer_idx: int,
        paging: PagedBatch,
    ) -> torch.Tensor:
        """Writes this call's K/V into the pool slots `paging` resolved, then one kernel
        launch attends every entry through its own block table."""
        q, k, v = self._project_qkv(normed, position_embeddings, layer_idx)
        k_pool, v_pool = self._k_pool[layer_idx], self._v_pool[layer_idx]
        k_pool[paging.dest_block_id, paging.dest_within] = k[0].transpose(0, 1)
        v_pool[paging.dest_block_id, paging.dest_within] = v[0].transpose(0, 1)
        out = paged_attention_forward(
            q[0], k_pool, v_pool, paging.block_table, paging.context_len, paging.query_offset,
            paging.q_start, paging.q_len, self._block_size, paging.max_q_len,
        )  # (n_heads, total_tokens, head_dim)
        out = out.transpose(0, 1).reshape(1, normed.shape[1], self.n_heads * self.head_dim)
        return self._layers[layer_idx].self_attn.o_proj(out)

    def contiguous_attention(
        self,
        normed: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        layer_idx: int,
        plan: BatchPlan,
        seqs: dict[int, Sequence],
        offsets: list[tuple[int, int]],
    ) -> torch.Tensor:
        """One kernel launch per entry, each over that sequence's own K/V buffer, so an
        entry's queries only ever see its own sequence."""
        q, k, v = self._project_qkv(normed, position_embeddings, layer_idx)
        total_tokens = normed.shape[1]
        cache = self._kv_cache.setdefault(layer_idx, {})
        out = torch.empty(
            1, total_tokens, self.n_heads, self.head_dim, device=normed.device, dtype=normed.dtype
        )
        n_groups = self.n_heads // self.n_kv_heads

        for entry, (start, end) in zip(plan, offsets, strict=True):
            n_new = end - start
            buf = cache.get(entry.seq_id)
            if buf is None:
                # Sized once for the sequence's whole lifetime and filled in place, so a
                # decode step copies only its own new K/V.
                seq = seqs[entry.seq_id]
                capacity = len(seq.prompt_tokens) + seq.sampling_params.max_tokens
                k_buf = torch.empty(
                    1, self.n_kv_heads, capacity, self.head_dim, device=k.device, dtype=k.dtype
                )
                v_buf = torch.empty_like(k_buf)
                filled = 0
            else:
                k_buf, v_buf, filled = buf

            k_buf[:, :, filled : filled + n_new] = k[:, :, start:end]
            v_buf[:, :, filled : filled + n_new] = v[:, :, start:end]
            filled += n_new
            cache[entry.seq_id] = (k_buf, v_buf, filled)

            # The kernel's causal mask places query row r at position query_offset + r.
            seq_k = k_buf[:, :, :filled].repeat_interleave(n_groups, dim=1)
            seq_v = v_buf[:, :, :filled].repeat_interleave(n_groups, dim=1)
            entry_out, _ = flash_attention_forward(
                q[:, :, start:end], seq_k, seq_v, is_causal=True, query_offset=filled - n_new
            )
            out[:, start:end] = entry_out.transpose(1, 2)

        out = out.reshape(1, total_tokens, self.n_heads * self.head_dim)
        return self._layers[layer_idx].self_attn.o_proj(out)

    def free(self, seq_id: int) -> None:
        """Drops the per-sequence state of a finished or preempted sequence. Pool blocks
        need nothing here: the allocator already reclaimed them."""
        self._past_key_values.pop(seq_id, None)
        for layer_cache in self._kv_cache.values():
            layer_cache.pop(seq_id, None)
        self._decode_graphs.forget(seq_id)

    @property
    def _layers(self) -> torch.nn.ModuleList:
        """Llama-family layers: self_attn, mlp, input_layernorm, post_attention_layernorm."""
        return self.model.model.layers


def eos_token_ids(eos: int | list[int] | None) -> frozenset[int]:
    """Instruct models can list several stop tokens (Llama-3: <|eot_id|> and
    <|end_of_text|>); any one of them finishes a sequence."""
    if eos is None:
        return frozenset()
    return frozenset(eos) if isinstance(eos, list) else frozenset({eos})


def _entry_token_ids(entry: BatchEntry, seq: Sequence) -> list[int]:
    """A prefill chunk's tokens (scheduler_step already advanced prefill_progress past
    them), or a decode step's input: the latest generated token, whose K/V it writes."""
    if entry.is_prefill_chunk:
        return seq.prefill_token_ids(seq.prefill_progress - entry.n_tokens, seq.prefill_progress)
    return [seq.generated_tokens[-1]]


def _owes_token(entry: BatchEntry, seq: Sequence) -> bool:
    return not (entry.is_prefill_chunk and seq.status == "PREFILLING")
