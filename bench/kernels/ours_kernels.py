"""The engine's own kernels, called the way the engine calls them. Nothing here re-implements
an operation: attention is the Triton kernel from `llm_serving_engine.kernels`, and the
projections are the separate `nn.Linear` calls the Llama layers make."""

from __future__ import annotations

import math
from typing import Callable

import torch

from llm_serving_engine.kernels.flash_attention import (
    decode_splits,
    flash_attention_forward,
    paged_attention_decode_forward,
)

from ..modelspec import ModelSpec
from .inputs import (
    DEVICE,
    decode_inputs,
    decode_reference,
    max_abs_error,
    prefill_inputs,
    prefill_reference,
    rand,
)
from .protocol import gemm_shape

BLOCK_SIZE = 16


def paged_decode(batch: int, ctx: int, spec: ModelSpec) -> tuple[Callable, torch.Tensor,
                                                                torch.Tensor, torch.Tensor]:
    """The engine's decode-attention call over a block table built from the canonical tensors,
    and those tensors."""
    q, k, v = decode_inputs(batch, ctx, spec)
    blocks = math.ceil(ctx / BLOCK_SIZE)
    padded = blocks * BLOCK_SIZE

    def pool(dense: torch.Tensor) -> torch.Tensor:
        rows = torch.zeros(batch, padded, spec.n_kv_heads, spec.head_dim, device=DEVICE,
                           dtype=dense.dtype)
        rows[:, :ctx] = dense
        return rows.view(batch * blocks, BLOCK_SIZE, spec.n_kv_heads, spec.head_dim)

    k_pool, v_pool = pool(k), pool(v)
    table = torch.arange(batch * blocks, dtype=torch.int32, device=DEVICE).view(batch, blocks)
    context_len = torch.full((batch,), ctx, dtype=torch.int32, device=DEVICE)
    q_start = torch.arange(batch, dtype=torch.int32, device=DEVICE)
    q_len = torch.ones(batch, dtype=torch.int32, device=DEVICE)
    q_heads_first = q.transpose(0, 1).contiguous()  # (n_heads, batch, head_dim)

    n_splits = decode_splits(batch)

    def call() -> torch.Tensor:
        return paged_attention_decode_forward(q_heads_first, k_pool, v_pool, table, context_len,
                                              q_start, q_len, BLOCK_SIZE, n_splits)

    return call, q, k, v


class OursKernels:
    name = "ours"
    notes = ("paged decode attention over a block table and causal prefill attention, one layer; "
             "excludes QKV projection, RoPE, the KV write and o_proj. Projections run as the "
             "engine runs them: separate q, k, v (and gate, up) linears")

    def decode_attention(self, batch: int, ctx: int, spec: ModelSpec) -> Callable | None:
        return paged_decode(batch, ctx, spec)[0]

    def decode_error(self, batch: int, ctx: int, spec: ModelSpec) -> float | None:
        call, q, k, v = paged_decode(batch, ctx, spec)
        return max_abs_error(call().transpose(0, 1), decode_reference(q, k, v))

    def prefill_attention(self, seq_len: int, spec: ModelSpec) -> Callable | None:
        q, k, v = (t.transpose(0, 1)[None].contiguous() for t in prefill_inputs(seq_len, spec))
        return lambda: flash_attention_forward(q, k, v, is_causal=True)

    def prefill_error(self, seq_len: int, spec: ModelSpec) -> float | None:
        q, k, v = prefill_inputs(seq_len, spec)
        out = self.prefill_attention(seq_len, spec)()[0].transpose(0, 1)
        return max_abs_error(out, prefill_reference(q, k, v))

    def gemm(self, which: str, m: int, spec: ModelSpec) -> Callable | None:
        n, k = gemm_shape(which, spec)
        x = rand(m, k, seed=7)
        widths = {
            "qkv_proj": [spec.n_heads * spec.head_dim, spec.n_kv_heads * spec.head_dim,
                         spec.n_kv_heads * spec.head_dim],
            "gate_up_proj": [spec.f, spec.f],
        }.get(which, [n])
        weights = [rand(width, k, seed=10 + i) for i, width in enumerate(widths)]
        return lambda: [torch.nn.functional.linear(x, w) for w in weights][-1]
