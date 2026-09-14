"""FlashAttention-2 in Triton: tiled attention with online softmax.

Q may be shorter than K/V for incremental decoding (a KV cache holds `query_offset`
prior tokens). The causal mask compares each query's true position, `query_offset + row`,
against each key position, so Q and K/V need not match in length. The backward pass
assumes query_offset=0: this engine never trains against a cache.

Notation follows the FlashAttention-2 paper: Q, K, V, O for the tensors, L for
the saved log-sum-exp, m and l for the running max and denominator.

    from llm_serving_engine.kernels.flash_attention import flash_attention_forward
    out, L = flash_attention_forward(Q, K, V, is_causal=True, query_offset=0)

`paged_attention_2`/`paged_attention_forward` is the batched, inference-only sibling:
one launch covers every entry in a BatchPlan against a shared per-layer K/V pool,
indexed by physical block id.
"""

import math

import torch
import triton
import triton.language as tl


def naive_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                    is_causal: bool = False, query_offset: int = 0) -> torch.Tensor:
    """Q: (batch, n_heads, seq_q, head_dim); K, V: (batch, n_heads, seq_k, head_dim)."""
    d = Q.shape[-1]
    S = (Q @ K.transpose(-1, -2)) / math.sqrt(d)
    if is_causal:
        nq, nk = Q.shape[-2], K.shape[-2]
        row = torch.arange(nq, device=Q.device)[:, None] + query_offset
        col = torch.arange(nk, device=Q.device)[None, :]
        S = S.masked_fill(row < col, float('-inf'))
    return torch.softmax(S, dim=-1) @ V

# Grid: (N_Q // BLOCK_M, Z * H). Z=batches, H=heads, N_Q=query positions,
# N_K=key/value positions (cache + new).
@triton.jit
def flash_attention_2(
    Q, K, V, O,
    L,
    stride_qz, stride_qh, stride_qm, stride_qd,
    stride_kz, stride_kh, stride_kn, stride_kd,
    stride_vz, stride_vh, stride_vn, stride_vd,
    stride_oz, stride_oh, stride_om, stride_od,
    Z, H, N_Q, N_K,
    D,                       # true head dimension (BLOCK_D is it rounded up)
    softmax_scale,
    QUERY_OFFSET,            # cached tokens before Q's first row (0 for fresh self-attention)
    BLOCK_M: tl.constexpr,   # query rows
    BLOCK_N: tl.constexpr,   # KVs
    BLOCK_D: tl.constexpr,   # head dim, rounded up to a power of 2
    IS_CAUSAL: tl.constexpr,
    PRECISION: tl.constexpr, # "ieee" for true fp32, "tf32" for tensor cores
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    off_z = off_hz // H
    off_h = off_hz % H

    q_offset = off_z * stride_qz + off_h * stride_qh
    k_offset = off_z * stride_kz + off_h * stride_kh
    v_offset = off_z * stride_vz + off_h * stride_vh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)

    m_mask = offs_m < N_Q

    offs_d = tl.arange(0, BLOCK_D)
    # BLOCK_D is D rounded up to a power of 2; lanes in [D, BLOCK_D) address the
    # next row's memory, so every load is masked and zero-filled -- contributes
    # nothing to any dot.
    d_mask = offs_d < D

    q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0)

    running_max = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    running_denom = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # A query row's true position is QUERY_OFFSET + offs_m, which bounds the furthest
    # key tile it can attend to; capped at N_K since the offset can push it past the cache.
    end_n = tl.minimum(QUERY_OFFSET + (start_m + 1) * BLOCK_M, N_K) if IS_CAUSAL else N_K

    for start_n in range(0, end_n, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N_K

        k_ptrs = K + k_offset + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)

        v_ptrs = V + v_offset + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)

        # softmax_scale is 1/sqrt(D), the true head dimension: BLOCK_D is only the padded
        # width, and scaling by it would shift the temperature whenever D isn't a power of 2.
        s = tl.dot(q, tl.trans(k), input_precision=PRECISION) * softmax_scale

        score_mask = m_mask[:, None] & n_mask[None, :]

        if IS_CAUSAL:
            # offs_m is a row within this Q tile; QUERY_OFFSET + offs_m is its
            # true position in the full (cache + new) sequence, which is what
            # offs_n must be compared against.
            score_mask = score_mask & (QUERY_OFFSET + offs_m[:, None] >= offs_n[None, :])

        s = tl.where(score_mask, s, float('-inf'))

        row_max = tl.max(s, axis=1)
        new_max = tl.maximum(running_max, row_max)

        alpha = tl.exp(running_max - new_max)

        p = tl.exp(s - new_max[:, None]) # softmax numerator for this block

        running_denom = alpha * running_denom + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, input_precision=PRECISION)

        running_max = new_max

    acc = acc / running_denom[:, None]

    L_ptrs = L + off_hz * N_Q + offs_m
    tl.store(L_ptrs, running_max + tl.log(running_denom), mask=m_mask)

    o_ptrs = O + off_z * stride_oz + off_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc, mask=m_mask[:, None] & d_mask[None, :])

# Tile schedule. fp16/bf16 feed tensor cores directly and fit the large tiles. True IEEE
# fp32 needs ~2x the shared memory for the same tile, because Triton emits a 3-pass
# split-float emulation for it, so fp32 trades tile size for accuracy. On an RTX 4070 Ti
# (99 KB shared/SM), fp32 at 128x64 needs 131 KB and does not launch.
_TILES = {
    torch.float16:  (128, 64),
    torch.bfloat16: (128, 64),
    torch.float32:  (64, 32),
}


def _softmax_precision(dtype: torch.dtype) -> str:
    """fp32 defaults to TF32 tensor cores in Triton, which costs ~3 decimal
    digits. Ask for IEEE explicitly so the fp32 path means fp32."""
    return "ieee" if dtype == torch.float32 else "tf32"


def _oom_hint(
    e: triton.runtime.errors.OutOfResources, D: int, dtype: torch.dtype, BLOCK_M: int, BLOCK_N: int
) -> triton.runtime.errors.OutOfResources:
    # Triton's message gives only byte counts; the cause is a (dtype, head_dim) tile that
    # doesn't fit in this GPU's shared memory.
    return triton.runtime.errors.OutOfResources(
        e.required, e.limit,
        f"shared memory for head_dim={D} in {dtype} at "
        f"BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N}. Use fp16/bf16, a smaller "
        f"head_dim, or a GPU with more shared memory per SM"
    )


def flash_attention_forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                            is_causal: bool = False, query_offset: int = 0):
    """Q: (Z, H, N_Q, D); K, V: (Z, H, N_K, D). N_Q and N_K may differ: a decode step
    passes N_Q=1 against the sequence's full cached N_K, with query_offset=N_K-N_Q so the
    causal mask compares true sequence positions.
    """
    Z, H, N_Q, D = q.shape
    Zk, Hk, N_K, Dk = k.shape
    assert (Z, H, D) == (Zk, Hk, Dk), "Q and K/V must share batch, heads, and head_dim"
    assert k.shape == v.shape, "K, V shapes don't match"
    assert q.dtype == k.dtype == v.dtype, "Q, K, V must share a dtype"
    assert q.dtype in _TILES, (
        f"unsupported dtype {q.dtype}; expected one of {tuple(_TILES)}"
    )
    assert q.is_cuda and k.is_cuda and v.is_cuda, "Q, K, V not all CUDA tensors"
    assert query_offset + N_Q <= N_K or not is_causal, (
        f"causal attention needs every query position covered by the cache: "
        f"query_offset({query_offset}) + N_Q({N_Q}) > N_K({N_K})"
    )

    o = torch.empty_like(q)
    L = torch.empty((Z, H, N_Q), device=q.device, dtype=torch.float32)

    BLOCK_M, BLOCK_N = _TILES[q.dtype]
    BLOCK_D = triton.next_power_of_2(D)

    grid = (triton.cdiv(N_Q, BLOCK_M), Z * H)

    try:
        flash_attention_2[grid](
            q, k, v, o, L,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            Z, H, N_Q, N_K,
            D,
            1.0 / math.sqrt(D),
            query_offset,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_D=BLOCK_D,
            IS_CAUSAL=is_causal,
            PRECISION=_softmax_precision(q.dtype),
        )
    except triton.runtime.errors.OutOfResources as e:
        raise _oom_hint(e, D, q.dtype, BLOCK_M, BLOCK_N) from None

    return o, L


@triton.jit
def paged_attention_2(
    Q, K_POOL, V_POOL, O,
    BLOCK_TABLE, CONTEXT_LEN, QUERY_OFFSET_ARR, Q_START, Q_LEN,
    stride_qh, stride_qm, stride_qd,
    stride_kb, stride_ks, stride_kh, stride_kd,
    stride_vb, stride_vs, stride_vh, stride_vd,
    stride_oh, stride_om, stride_od,
    stride_bt_row,
    H, D,
    BLOCK_SIZE_KV,          # physical KV block size (runtime int, any positive value)
    softmax_scale,
    N_GROUPS: tl.constexpr,  # H // H_KV, query heads sharing one KV head under GQA
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    PRECISION: tl.constexpr,
):
    """One program per (query tile, batch entry, query head). Grid axis 0 is sized to the
    longest entry in the batch.

    K/V live in a shared per-layer pool indexed by physical block id. BLOCK_TABLE[entry, i]
    is the physical block backing that entry's cache positions [i*BLOCK_SIZE_KV,
    (i+1)*BLOCK_SIZE_KV); only positions below CONTEXT_LEN are ever read.
    """
    start_m = tl.program_id(0)
    entry_head = tl.program_id(1)
    entry_idx = entry_head // H
    head_idx = entry_head % H
    kv_head_idx = head_idx // N_GROUPS

    q_len = tl.load(Q_LEN + entry_idx)
    if start_m * BLOCK_M >= q_len:
        return

    q_start = tl.load(Q_START + entry_idx)
    context_len = tl.load(CONTEXT_LEN + entry_idx)
    query_offset = tl.load(QUERY_OFFSET_ARR + entry_idx)

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < q_len
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < D

    q_ptrs = Q + head_idx * stride_qh + (q_start + offs_m)[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0)

    running_max = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    running_denom = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # Same bound as flash_attention_2: a query tile need not scan past the furthest
    # position any of its rows can causally attend to.
    end_n = tl.minimum(query_offset + (start_m + 1) * BLOCK_M, context_len)

    for start_n in range(0, end_n, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < context_len

        block_in_table = offs_n // BLOCK_SIZE_KV
        within_block = offs_n % BLOCK_SIZE_KV
        block_id = tl.load(BLOCK_TABLE + entry_idx * stride_bt_row + block_in_table, mask=n_mask, other=0)

        k_ptrs = (K_POOL + block_id[:, None] * stride_kb + within_block[:, None] * stride_ks
                  + kv_head_idx * stride_kh + offs_d[None, :] * stride_kd)
        k = tl.load(k_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
        v_ptrs = (V_POOL + block_id[:, None] * stride_vb + within_block[:, None] * stride_vs
                  + kv_head_idx * stride_vh + offs_d[None, :] * stride_vd)
        v = tl.load(v_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)

        s = tl.dot(q, tl.trans(k), input_precision=PRECISION) * softmax_scale
        score_mask = m_mask[:, None] & n_mask[None, :]
        score_mask = score_mask & (query_offset + offs_m[:, None] >= offs_n[None, :])
        s = tl.where(score_mask, s, float('-inf'))

        row_max = tl.max(s, axis=1)
        new_max = tl.maximum(running_max, row_max)
        alpha = tl.exp(running_max - new_max)
        p = tl.exp(s - new_max[:, None])
        running_denom = alpha * running_denom + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, input_precision=PRECISION)
        running_max = new_max

    acc = acc / running_denom[:, None]
    o_ptrs = O + head_idx * stride_oh + (q_start + offs_m)[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc, mask=m_mask[:, None] & d_mask[None, :])


def paged_attention_forward(
    q: torch.Tensor, k_pool: torch.Tensor, v_pool: torch.Tensor,
    block_table: torch.Tensor, context_len: torch.Tensor, query_offset: torch.Tensor,
    q_start: torch.Tensor, q_len: torch.Tensor, block_size: int, max_q_len: int,
) -> torch.Tensor:
    """One kernel launch for a whole BatchPlan's attention at one layer, against K/V
    stored in a shared paged pool.

    q: (H, total_tokens, D).
    k_pool/v_pool: (num_blocks, block_size, H_KV, D), shared across every entry and
    already holding this call's new tokens.
    block_table/context_len/query_offset/q_start/q_len: one row per batch entry.
    max_q_len: q_len's max.
    """
    H, _total_tokens, D = q.shape
    H_KV = k_pool.shape[2]
    assert q.is_cuda and k_pool.is_cuda and v_pool.is_cuda, "Q, K, V not all CUDA tensors"
    assert q.dtype == k_pool.dtype == v_pool.dtype, "Q, K, V must share a dtype"
    assert q.dtype in _TILES, f"unsupported dtype {q.dtype}; expected one of {tuple(_TILES)}"
    assert H % H_KV == 0, f"query heads ({H}) must be a multiple of KV heads ({H_KV})"

    o = torch.empty_like(q)
    BLOCK_M, BLOCK_N = _TILES[q.dtype]
    BLOCK_D = triton.next_power_of_2(D)
    grid = (triton.cdiv(max_q_len, BLOCK_M), block_table.shape[0] * H)

    try:
        paged_attention_2[grid](
            q, k_pool, v_pool, o,
            block_table, context_len, query_offset, q_start, q_len,
            q.stride(0), q.stride(1), q.stride(2),
            k_pool.stride(0), k_pool.stride(1), k_pool.stride(2), k_pool.stride(3),
            v_pool.stride(0), v_pool.stride(1), v_pool.stride(2), v_pool.stride(3),
            o.stride(0), o.stride(1), o.stride(2),
            block_table.stride(0),
            H, D,
            block_size,
            1.0 / math.sqrt(D),
            N_GROUPS=H // H_KV,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
            PRECISION=_softmax_precision(q.dtype),
        )
    except triton.runtime.errors.OutOfResources as e:
        raise _oom_hint(e, D, q.dtype, BLOCK_M, BLOCK_N) from None

    return o


def flash_attention_backward(Q, K, V, O, L, dO, is_causal, scale):
    """Training has no KV cache, so query_offset is always 0 here."""
    S = (Q @ K.transpose(-1, -2)) * scale
    if is_causal:
        nq, nk = Q.shape[-2], K.shape[-2]
        mask = torch.tril(torch.ones(nq, nk, dtype=torch.bool, device=Q.device))
        S = S.masked_fill(~mask, -1e6)
    P = torch.exp(S - L.unsqueeze(-1))
    dV = P.transpose(-1, -2) @ dO
    dP = dO @ V.transpose(-1, -2)
    Dv = (dO * O).sum(dim=-1, keepdim=True)
    dS = P * (dP - Dv)
    dQ = (dS @ K) * scale
    dK = (dS.transpose(-1, -2) @ Q) * scale
    return dQ, dK, dV


class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        O, L = flash_attention_forward(Q, K, V, is_causal)
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        ctx.scale = 1.0 / math.sqrt(Q.shape[-1])
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        dQ, dK, dV = flash_attention_backward(Q, K, V, O, L, dO.contiguous(),
                                              ctx.is_causal, ctx.scale)
        return dQ, dK, dV, None
