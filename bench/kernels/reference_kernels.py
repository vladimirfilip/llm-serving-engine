"""Reference contenders: torch, FlashAttention and FlashInfer. A library that is not installed
or does not run on this GPU is reported as unavailable, not silently dropped."""

from __future__ import annotations

import math
from importlib import metadata
from typing import Callable

import torch

from ..modelspec import ModelSpec
from .inputs import (
    DEVICE,
    DTYPE,
    decode_inputs,
    decode_reference,
    max_abs_error,
    prefill_inputs,
    prefill_reference,
    rand,
)
from .ours_kernels import BLOCK_SIZE
from .protocol import gemm_shape


def installed_version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


class TorchKernels:
    name = "torch"
    notes = ("F.linear as one fused matmul per projection (cuBLAS or cuBLASLt, whichever torch "
             "picks); fp32 scaled_dot_product_attention is the numeric reference, not a timed "
             "contender")

    def decode_attention(self, batch: int, ctx: int, spec: ModelSpec) -> Callable | None:
        return None

    def prefill_attention(self, seq_len: int, spec: ModelSpec) -> Callable | None:
        return None

    def decode_error(self, batch: int, ctx: int, spec: ModelSpec) -> float | None:
        return None

    def prefill_error(self, seq_len: int, spec: ModelSpec) -> float | None:
        return None

    def gemm(self, which: str, m: int, spec: ModelSpec) -> Callable | None:
        n, k = gemm_shape(which, spec)
        x, w = rand(m, k, seed=7), rand(n, k, seed=11)
        return lambda: torch.nn.functional.linear(x, w)


class FlashAttnKernels:
    name = "flash_attn"

    def __init__(self):
        import flash_attn

        self.flash_attn = flash_attn
        self.version = flash_attn.__version__
        self.notes = f"flash-attn {self.version}: attention only, one layer, dense KV cache"

    def decode_attention(self, batch: int, ctx: int, spec: ModelSpec) -> Callable | None:
        q, k, v = decode_inputs(batch, ctx, spec)
        lens = torch.full((batch,), ctx, dtype=torch.int32, device=DEVICE)
        q = q[:, None]
        return lambda: self.flash_attn.flash_attn_with_kvcache(q, k, v, cache_seqlens=lens,
                                                               causal=True)

    def decode_error(self, batch: int, ctx: int, spec: ModelSpec) -> float | None:
        q, k, v = decode_inputs(batch, ctx, spec)
        return max_abs_error(self.decode_attention(batch, ctx, spec)()[:, 0],
                              decode_reference(q, k, v))

    def prefill_attention(self, seq_len: int, spec: ModelSpec) -> Callable | None:
        q, k, v = (t[None] for t in prefill_inputs(seq_len, spec))
        return lambda: self.flash_attn.flash_attn_func(q, k, v, causal=True)

    def prefill_error(self, seq_len: int, spec: ModelSpec) -> float | None:
        q, k, v = prefill_inputs(seq_len, spec)
        return max_abs_error(self.prefill_attention(seq_len, spec)()[0],
                             prefill_reference(q, k, v))

    def gemm(self, which: str, m: int, spec: ModelSpec) -> Callable | None:
        return None


class FlashInferKernels:
    name = "flashinfer"

    def __init__(self):
        import flashinfer

        self.flashinfer = flashinfer
        self.version = flashinfer.__version__
        self.notes = (f"flashinfer {self.version}: attention only, one layer; decode over paged "
                      f"KV with page size {BLOCK_SIZE}")

    def decode_attention(self, batch: int, ctx: int, spec: ModelSpec) -> Callable | None:
        q, k, v = decode_inputs(batch, ctx, spec)
        pages = math.ceil(ctx / BLOCK_SIZE)
        padded = pages * BLOCK_SIZE
        kv = torch.zeros(batch * pages, 2, BLOCK_SIZE, spec.n_kv_heads, spec.head_dim,
                         device=DEVICE, dtype=DTYPE)
        for which, dense in enumerate((k, v)):
            rows = torch.zeros(batch, padded, spec.n_kv_heads, spec.head_dim, device=DEVICE,
                               dtype=DTYPE)
            rows[:, :ctx] = dense
            kv[:, which] = rows.view(batch * pages, BLOCK_SIZE, spec.n_kv_heads, spec.head_dim)
        workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)
        wrapper = self.flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, "NHD")
        indptr = torch.arange(0, (batch + 1) * pages, pages, dtype=torch.int32, device=DEVICE)
        indices = torch.arange(batch * pages, dtype=torch.int32, device=DEVICE)
        last = torch.full((batch,), ctx - (pages - 1) * BLOCK_SIZE, dtype=torch.int32,
                          device=DEVICE)
        wrapper.plan(indptr, indices, last, spec.n_heads, spec.n_kv_heads, spec.head_dim,
                     BLOCK_SIZE, data_type=DTYPE, q_data_type=DTYPE)
        return lambda: wrapper.run(q, kv)

    def decode_error(self, batch: int, ctx: int, spec: ModelSpec) -> float | None:
        q, k, v = decode_inputs(batch, ctx, spec)
        return max_abs_error(self.decode_attention(batch, ctx, spec)(), decode_reference(q, k, v))

    def prefill_attention(self, seq_len: int, spec: ModelSpec) -> Callable | None:
        q, k, v = prefill_inputs(seq_len, spec)
        return lambda: self.flashinfer.single_prefill_with_kv_cache(q, k, v, causal=True)

    def prefill_error(self, seq_len: int, spec: ModelSpec) -> float | None:
        q, k, v = prefill_inputs(seq_len, spec)
        return max_abs_error(self.prefill_attention(seq_len, spec)(), prefill_reference(q, k, v))

    def gemm(self, which: str, m: int, spec: ModelSpec) -> Callable | None:
        return None


def available_contenders() -> tuple[list, dict[str, str]]:
    """The reference adapters that import, and why each other one does not."""
    found, missing = [TorchKernels()], {}
    for adapter_class, distribution in ((FlashAttnKernels, "flash-attn"),
                                        (FlashInferKernels, "flashinfer-python")):
        if installed_version(distribution) is None:
            missing[adapter_class.name] = f"{distribution} is not installed"
            continue
        try:
            found.append(adapter_class())
        except Exception as e:  # a broken install must be named, not skipped
            missing[adapter_class.name] = f"{distribution} failed to import: {e}"
    return found, missing
