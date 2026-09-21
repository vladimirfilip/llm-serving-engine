"""Canonical attention inputs and the fp32 reference every contender is checked against. Each
adapter derives its own layout (paged pool, block table, NHD pages) from these tensors, so
they all compute the same attention."""

from __future__ import annotations

import torch

from ..modelspec import ModelSpec

DEVICE = "cuda"
DTYPE = torch.bfloat16
ATTENTION_TOLERANCE = 2e-2


def rand(*shape: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    return torch.randn(*shape, device=DEVICE, dtype=DTYPE, generator=gen)


def decode_inputs(batch: int, ctx: int, spec: ModelSpec) -> tuple[torch.Tensor, ...]:
    """q (batch, n_heads, head_dim); k, v (batch, ctx, n_kv_heads, head_dim)."""
    return (rand(batch, spec.n_heads, spec.head_dim, seed=3),
            rand(batch, ctx, spec.n_kv_heads, spec.head_dim, seed=1),
            rand(batch, ctx, spec.n_kv_heads, spec.head_dim, seed=2))


def prefill_inputs(seq_len: int, spec: ModelSpec) -> tuple[torch.Tensor, ...]:
    """q (seq_len, n_heads, head_dim); k, v (seq_len, n_kv_heads, head_dim)."""
    return (rand(seq_len, spec.n_heads, spec.head_dim, seed=4),
            rand(seq_len, spec.n_kv_heads, spec.head_dim, seed=5),
            rand(seq_len, spec.n_kv_heads, spec.head_dim, seed=6))


def sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool) -> torch.Tensor:
    """fp32 scaled dot-product attention with grouped K/V heads. q (H, Nq, D); k, v (Hkv, Nk, D)."""
    out = torch.nn.functional.scaled_dot_product_attention(
        q[None].float(), k[None].float(), v[None].float(), is_causal=causal, enable_gqa=True
    )
    return out[0]


def decode_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """(batch, n_heads, head_dim): each sequence's one query attends to all of its context."""
    return torch.stack([
        sdpa(q[b][:, None], k[b].transpose(0, 1), v[b].transpose(0, 1), causal=False)[:, 0]
        for b in range(q.shape[0])
    ])


def prefill_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """(seq_len, n_heads, head_dim), causal."""
    out = sdpa(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), causal=True)
    return out.transpose(0, 1)


def max_abs_error(out: torch.Tensor, reference: torch.Tensor) -> float:
    return float((out.float() - reference).abs().max())
