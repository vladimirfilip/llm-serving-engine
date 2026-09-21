"""What a kernel contender provides. Each method allocates its inputs, then returns a
zero-argument closure that runs exactly what the engine runs for that operation, or None when
the contender does not support the combination."""

from __future__ import annotations

from typing import Callable, Protocol

import torch

from ..modelspec import ModelSpec

GEMM_SHAPES = ("qkv_proj", "o_proj", "gate_up_proj", "down_proj", "lm_head")


class KernelAdapter(Protocol):
    name: str
    notes: str  # the operations the closure includes and excludes

    def decode_attention(
        self, batch: int, ctx: int, spec: ModelSpec
    ) -> Callable[[], torch.Tensor] | None: ...

    def prefill_attention(
        self, seq_len: int, spec: ModelSpec
    ) -> Callable[[], torch.Tensor] | None: ...

    def gemm(self, which: str, m: int, spec: ModelSpec) -> Callable[[], torch.Tensor] | None: ...

    def decode_error(self, batch: int, ctx: int, spec: ModelSpec) -> float | None:
        """Largest absolute difference from the fp32 reference, or None when unsupported."""

    def prefill_error(self, seq_len: int, spec: ModelSpec) -> float | None: ...


def gemm_shape(which: str, spec: ModelSpec) -> tuple[int, int]:
    """(N, K) of a projection: output width and input width."""
    return {
        "qkv_proj": ((spec.n_heads + 2 * spec.n_kv_heads) * spec.head_dim, spec.d),
        "o_proj": (spec.d, spec.n_heads * spec.head_dim),
        "gate_up_proj": (2 * spec.f, spec.d),
        "down_proj": (spec.d, spec.f),
        "lm_head": (spec.vocab, spec.d),
    }[which]
