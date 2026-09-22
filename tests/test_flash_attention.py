"""paged_attention_decode_forward against a straightforward fp32 reference: ragged
contexts, GQA, a block size that isn't a power of 2, a context shorter than one split,
a graph bucket's padding rows, and workspace reuse across calls.
"""

from __future__ import annotations

import pytest
import torch

from llm_serving_engine.kernels.flash_attention import (
    decode_workspace,
    paged_attention_decode_forward,
)

pytestmark = pytest.mark.cuda

DEVICE = "cuda"
# fp32's true error is ~1e-6 here; 1e-4 still leaves ~100x headroom while catching a
# wrong key or a split-boundary off-by-one, which bf16's own rounding (its actual error,
# ~4e-3) would otherwise hide well past a merely loose bf16 tolerance.
FP32_TOLERANCE = 1e-4
BF16_TOLERANCE = 2e-2


def decode_case(
    context_lens: list[int], n_heads: int, n_kv_heads: int, head_dim: int, block_size: int,
    dtype: torch.dtype = torch.float32, pad_rows: int = 0,
):
    """A paged K/V pool and block tables for one decode-only batch: `len(context_lens)`
    real rows (query offset = context_len - 1) plus `pad_rows` graph-padding rows (q_len
    0, pointing at a shared scratch block never read by a real row)."""
    torch.manual_seed(0)
    num_entries = len(context_lens)
    total_rows = num_entries + pad_rows
    max_blocks = max((c + block_size - 1) // block_size for c in context_lens)
    num_blocks = num_entries * max_blocks + 1
    scratch = num_blocks - 1

    k_pool = torch.zeros(num_blocks, block_size, n_kv_heads, head_dim, device=DEVICE, dtype=dtype)
    v_pool = torch.zeros(num_blocks, block_size, n_kv_heads, head_dim, device=DEVICE, dtype=dtype)
    block_table = torch.full((total_rows, max_blocks), scratch, dtype=torch.int32, device=DEVICE)
    k_by_entry, v_by_entry = [], []
    for e, ctx in enumerate(context_lens):
        k = torch.randn(ctx, n_kv_heads, head_dim, device=DEVICE, dtype=dtype)
        v = torch.randn(ctx, n_kv_heads, head_dim, device=DEVICE, dtype=dtype)
        k_by_entry.append(k)
        v_by_entry.append(v)
        for b in range((ctx + block_size - 1) // block_size):
            block_table[e, b] = e * max_blocks + b
            lo, hi = b * block_size, min((b + 1) * block_size, ctx)
            k_pool[e * max_blocks + b, : hi - lo] = k[lo:hi]
            v_pool[e * max_blocks + b, : hi - lo] = v[lo:hi]

    q = torch.randn(n_heads, total_rows, head_dim, device=DEVICE, dtype=dtype)
    context_len = torch.tensor(context_lens + [1] * pad_rows, dtype=torch.int32, device=DEVICE)
    q_start = torch.arange(total_rows, dtype=torch.int32, device=DEVICE)
    q_len = torch.tensor([1] * num_entries + [0] * pad_rows, dtype=torch.int32, device=DEVICE)

    def reference() -> torch.Tensor:
        """(n_heads, num_entries, head_dim), fp32, one query row per entry against its
        own context -- torch's own attention, the numeric ground truth every kernel in
        this codebase is checked against."""
        n_groups = n_heads // n_kv_heads
        out = torch.empty(n_heads, num_entries, head_dim, device=DEVICE, dtype=torch.float32)
        for e in range(num_entries):
            qv = q[:, e].float().view(n_kv_heads, n_groups, 1, head_dim)
            k = k_by_entry[e].float().transpose(0, 1)[:, None].expand(-1, n_groups, -1, -1)
            v = v_by_entry[e].float().transpose(0, 1)[:, None].expand(-1, n_groups, -1, -1)
            attn = torch.nn.functional.scaled_dot_product_attention(qv, k, v)
            out[:, e] = attn.reshape(n_heads, head_dim)
        return out

    return q, k_pool, v_pool, block_table, context_len, q_start, q_len, reference


@pytest.mark.parametrize(
    "name,context_lens,n_heads,n_kv_heads,head_dim,block_size,n_splits",
    [
        ("single split", [40], 8, 8, 64, 16, 1),
        ("split doesn't divide context evenly", [40], 8, 8, 64, 16, 3),
        ("ragged contexts, one head per kv head", [1, 17, 200, 33, 999], 8, 8, 64, 16, 8),
        ("gqa", [100, 250, 5], 24, 8, 128, 16, 8),
        ("context shorter than one split", [1, 3], 4, 4, 32, 16, 16),
        ("non-power-of-2 block size", [7, 44, 130], 8, 8, 64, 13, 6),
        ("llama-3.2-3b head config", [16000, 1, 8192, 500, 100, 3], 24, 8, 128, 16, 16),
    ],
)
def test_matches_reference_attention(
    name, context_lens, n_heads, n_kv_heads, head_dim, block_size, n_splits
):
    q, k_pool, v_pool, block_table, context_len, q_start, q_len, reference = decode_case(
        context_lens, n_heads, n_kv_heads, head_dim, block_size
    )
    out = paged_attention_decode_forward(
        q, k_pool, v_pool, block_table, context_len, q_start, q_len, block_size, n_splits
    )
    err = (out.float() - reference()).abs().max().item()
    assert err < FP32_TOLERANCE, f"{name}: max_abs_error={err}"


def test_matches_reference_attention_in_bf16():
    q, k_pool, v_pool, block_table, context_len, q_start, q_len, reference = decode_case(
        [33, 33, 33, 33], n_heads=8, n_kv_heads=8, head_dim=64, block_size=16,
        dtype=torch.bfloat16,
    )
    out = paged_attention_decode_forward(
        q, k_pool, v_pool, block_table, context_len, q_start, q_len, block_size=16, n_splits=4
    )
    err = (out.float() - reference()).abs().max().item()
    assert err < BF16_TOLERANCE


def test_padding_rows_from_a_graph_bucket_do_not_corrupt_real_rows():
    """A decode graph's bucket pads every batch up to its size; padding rows (q_len 0)
    share one scratch block and must never poison a real row's output. Combine never
    writes a padding row's own output slot (nothing downstream reads it), so only the
    real rows are checked here."""
    q, k_pool, v_pool, block_table, context_len, q_start, q_len, reference = decode_case(
        [10, 500, 42], n_heads=8, n_kv_heads=8, head_dim=64, block_size=16, pad_rows=5
    )
    out = paged_attention_decode_forward(
        q, k_pool, v_pool, block_table, context_len, q_start, q_len, block_size=16, n_splits=8
    )
    real = out[:, :3, :].float()
    assert torch.isfinite(real).all(), "a padding row's neighbor produced inf/nan"
    err = (real - reference()).abs().max().item()
    assert err < FP32_TOLERANCE


def test_split_k_never_reads_an_unwritten_partial():
    """Every (split, entry, head) slot of the workspace must be written before combine
    reads it, an empty split's neutral partial included -- a short context leaves most
    splits empty (context_len=1 against 16 splits: 15 of them). Pre-filling the
    workspace with NaN, rather than relying on a fresh `torch.empty`'s allocator luck,
    catches a slot the kernel forgets to write every time, not just sometimes."""
    q, k_pool, v_pool, block_table, context_len, q_start, q_len, reference = decode_case(
        [1, 3, 40], n_heads=8, n_kv_heads=8, head_dim=64, block_size=16
    )
    n_splits = 16
    ws = decode_workspace(n_splits, num_entries=3, n_heads=8, head_dim=64, device=DEVICE)
    for t in ws:
        t.fill_(float("nan"))
    out = paged_attention_decode_forward(
        q, k_pool, v_pool, block_table, context_len, q_start, q_len, block_size=16,
        n_splits=n_splits, workspace=ws,
    )
    assert torch.isfinite(out).all(), "an unwritten workspace slot leaked into the output"
    err = (out.float() - reference()).abs().max().item()
    assert err < FP32_TOLERANCE


def test_the_explicit_workspace_matches_an_internally_allocated_one_across_reuse():
    """A CUDA graph's workspace is the same tensors every replay, holding the previous
    replay's partials until this one overwrites them. Reusing one workspace across two
    calls, the second with shorter contexts (more empty splits reading over what the
    first call left behind), must still match eager's own fresh allocation each time --
    not just on a workspace's first, still-empty use."""
    ws = decode_workspace(n_splits=8, num_entries=4, n_heads=8, head_dim=64, device=DEVICE)
    for context_lens in ([33, 33, 33, 33], [2, 900, 1, 50]):
        q, k_pool, v_pool, block_table, context_len, q_start, q_len, _reference = decode_case(
            context_lens, n_heads=8, n_kv_heads=8, head_dim=64, block_size=16
        )
        eager = paged_attention_decode_forward(
            q, k_pool, v_pool, block_table, context_len, q_start, q_len, block_size=16, n_splits=8
        )
        explicit = paged_attention_decode_forward(
            q, k_pool, v_pool, block_table, context_len, q_start, q_len, block_size=16,
            n_splits=8, workspace=ws,
        )
        assert torch.equal(eager, explicit)
