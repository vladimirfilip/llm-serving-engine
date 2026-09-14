"""Model loading and the GPU worker's per-iteration forward pass.

Two forward paths, selected by config.use_custom_kernels:

- Default (forward): each in-flight Sequence gets its own HuggingFace
  `past_key_values` cache keyed by seq_id, instead of one physical paged pool. Runs
  one HF call per BatchPlan entry.
- Fused (forward_fused): flattens the whole BatchPlan into one [1, total_tokens]
  tensor and runs the decoder stack manually, layer by layer, so attention_forward
  and ffn_forward each see the whole batch at once. attention_forward still loops
  per-sequence internally, since each sequence has its own K/V cache tensor;
  ffn_forward doesn't need to, since FFN is token-independent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .batch_plan import BatchEntry, BatchPlan
from .config import ModelConfig
from .sampling import SamplingParams
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
        self._kv_cache: dict[int, dict[int, tuple[Any, Any]]] = {}  # layer_idx -> seq_id -> (K, V)

        self.device = config.device if torch.cuda.is_available() else "cpu"
        dtype = getattr(torch, config.dtype)  # an unknown dtype name must fail loudly
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path, dtype=dtype
        ).to(self.device)
        self.model.eval()

        if config.use_custom_kernels:
            self.wire_custom_kernels()

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

    def build_tensors(self, entry: BatchEntry, seq: Sequence) -> torch.Tensor:
        """input_ids for one BatchPlan entry: the new slice of prompt tokens for a
        prefill chunk (seq.prefill_progress already advanced past it by scheduler_step),
        or the just-generated token for a decode step.

        Assumes handle_iteration_results has already appended each sequence's previous
        token before this runs, so generated_tokens[-1] is always the right decode input.
        """
        import torch

        if entry.is_prefill_chunk:
            token_ids = seq.prompt_tokens[seq.prefill_progress - entry.n_tokens : seq.prefill_progress]
        else:
            token_ids = [seq.generated_tokens[-1]]
        return torch.tensor([token_ids], device=self.device)

    def forward(self, plan: BatchPlan, seqs: dict[int, Sequence]) -> list[tuple[int, int, bool]]:
        """One GPU-worker iteration: run every BatchPlan entry against its own per-sequence
        KV cache. A prefill entry that doesn't finish the prompt this iteration (seq.status
        is still "PREFILLING" — scheduler_step already flips it to "DECODING" on the
        completing chunk) extends the cache but samples nothing: there's no token to report
        until the prompt is fully in cache, same as an unchunked prefill's first call.
        """
        import torch

        if self.config.use_custom_kernels:
            return self.forward_fused(plan, seqs)

        results = []
        for entry in plan:
            seq = seqs[entry.seq_id]
            input_ids = self.build_tensors(entry, seq)
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
        """Concatenates every BatchPlan entry's tokens into one [1, total_tokens] sequence —
        the flattened layout attention_forward and ffn_forward operate on. offsets[i] is the
        (start, end) span entry i occupies in that flattened tensor, in plan order.
        """
        import torch

        input_ids, position_ids, offsets = [], [], []
        cursor = 0
        for entry in plan:
            seq = seqs[entry.seq_id]
            input_ids.append(self.build_tensors(entry, seq))
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

    def attention_forward(
        self,
        normed: torch.Tensor,
        plan: BatchPlan,
        seqs: dict[int, Sequence],
        offsets: list[tuple[int, int]],
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        layer_idx: int,
    ) -> torch.Tensor:
        """FlashAttention (kernels/flash_attention.py), one sequence at a time against its
        own disjoint K/V cache tensor — the same per-sequence-buffer tradeoff the default
        HF path makes with `_past_key_values`. Unlike ffn_forward, this can't run as a
        single call over the flattened tensor: each entry's queries must only attend to
        its own sequence's cache.

        query_offset is however many tokens are already cached before this call's new
        ones; the kernel's causal mask compares each query's true position
        (query_offset + row) against each key position, so Q can be shorter than K/V — a
        decode step's single new token still attends to its whole cache, and a prefill
        chunk past the first still only sees causally-valid keys.
        """
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

        cache = self._kv_cache.setdefault(layer_idx, {})
        out = torch.empty(1, total_tokens, n_heads, head_dim, device=normed.device, dtype=normed.dtype)

        for entry, (start, end) in zip(plan, offsets, strict=True):
            seq_q = q[:, :, start:end]
            past = cache.get(entry.seq_id)
            seq_k = k[:, :, start:end] if past is None else torch.cat([past[0], k[:, :, start:end]], dim=2)
            seq_v = v[:, :, start:end] if past is None else torch.cat([past[1], v[:, :, start:end]], dim=2)
            cache[entry.seq_id] = (seq_k, seq_v)  # disjoint per-sequence buffer, not a shared pool

            query_offset = seq_k.shape[2] - seq_q.shape[2]
            seq_k = seq_k.repeat_interleave(n_groups, dim=1)  # GQA: broadcast kv heads onto q heads
            seq_v = seq_v.repeat_interleave(n_groups, dim=1)

            entry_out, _ = flash_attention_forward(
                seq_q, seq_k, seq_v, is_causal=True, query_offset=query_offset
            )
            out[:, start:end] = entry_out.transpose(1, 2)

        return attn.o_proj(out.reshape(1, total_tokens, n_heads * head_dim))

    def ffn_forward(self, normed: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Every plan token is independent in the FFN, so unlike attention_forward this
        runs as a single call over the whole flattened tensor. Calls the layer's `mlp`
        submodule directly, not the layer's own `forward` — that would rerun self-attention
        (without the paged cache or causal mask this engine needs) and double the residual
        add the caller already does.
        """
        return self._layers[layer_idx].mlp(normed)

    def forward_fused(self, plan: BatchPlan, seqs: dict[int, Sequence]) -> list[tuple[int, int, bool]]:
        """Fused GPU-worker iteration: embeds the whole BatchPlan once, runs
        attention_forward/ffn_forward per layer over the flattened tensor, then samples only
        the rows that owe a token — same eligibility rule as forward()'s per-entry loop
        (decode entries, and prefill entries whose chunk just completed the prompt).
        """
        import torch

        input_ids, position_ids, offsets = self._flatten_plan(plan, seqs)
        hidden = self.model.get_input_embeddings()(input_ids)
        # cos/sin depend only on position_ids and head_dim, not layer_idx — computed once
        # here rather than inside attention_forward's per-layer loop.
        position_embeddings = self.model.model.rotary_emb(hidden, position_ids)

        with torch.no_grad():
            for layer_idx, layer in enumerate(self._layers):
                attn_out = self.attention_forward(
                    layer.input_layernorm(hidden), plan, seqs, offsets, position_embeddings, layer_idx
                )
                hidden = hidden + attn_out
                ffn_out = self.ffn_forward(layer.post_attention_layernorm(hidden), layer_idx)
                hidden = hidden + ffn_out
            hidden = self.model.model.norm(hidden)

            results = []
            for entry, (_start, end) in zip(plan, offsets, strict=True):
                seq = seqs[entry.seq_id]
                if entry.is_prefill_chunk and seq.status == "PREFILLING":
                    continue
                logits = self.model.get_output_embeddings()(hidden[:, end - 1])
                token_id = self._sample(logits[0], seq.sampling_params)
                finished = token_id in self.eos_token_ids or (
                    len(seq.generated_tokens) + 1 >= seq.sampling_params.max_tokens
                )
                results.append((seq.seq_id, token_id, finished))
        return results

    def free(self, seq_id: int) -> None:
        """Drops a finished/cancelled sequence's KV cache. Call from result handling,
        alongside allocator.free, so this cache doesn't outlive the sequence it belongs to."""
        self._past_key_values.pop(seq_id, None)
        for layer_cache in self._kv_cache.values():
            layer_cache.pop(seq_id, None)

    def _sample(self, logits: torch.Tensor, params: SamplingParams) -> int:
        import torch

        if params.temperature == 0:
            return int(torch.argmax(logits).item())

        logits = logits / params.temperature
        if params.top_k > 0:
            top_k = min(params.top_k, logits.size(-1))
            kth_value = torch.topk(logits, top_k).values[..., -1]
            logits = torch.where(logits < kth_value, torch.full_like(logits, float("-inf")), logits)

        probs = torch.softmax(logits, dim=-1)
        if params.top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            drop = cumulative > params.top_p
            drop[..., 1:] = drop[..., :-1].clone()
            drop[..., 0] = False
            sorted_probs[drop] = 0.0
            probs = torch.zeros_like(probs).scatter(-1, sorted_idx, sorted_probs)
            probs = probs / probs.sum()

        return int(torch.multinomial(probs, 1).item())

    @property
    def eos_token_ids(self) -> frozenset[int]:
        # Instruct-tuned models (Llama-3's <|eot_id|> alongside <|end_of_text|>) list more
        # than one stop token; matching only the first silently never finishes on the rest.
        eos = self.model.generation_config.eos_token_id
        return frozenset(eos) if isinstance(eos, list) else frozenset({eos})
