"""FlashAttention-2 in Triton: tiled attention with online softmax.

Q may be shorter than K/V for incremental decoding (a KV cache holds `query_offset`
prior tokens). The causal mask compares each query's true position, `query_offset + row`,
against each key position, so Q and K/V need not match in length. Forward only: this
engine never trains.

Notation follows the FlashAttention-2 paper: Q, K, V, O for the tensors, m and l for the
running max and denominator. Under grouped-query attention, N_GROUPS query heads share
each K/V head; both kernels read the shared head in place.

    from llm_serving_engine.kernels.flash_attention import flash_attention_forward
    O = flash_attention_forward(Q, K, V, is_causal=True, query_offset=0)

`paged_attention_2`/`paged_attention_forward` is the batched, inference-only sibling:
one launch covers every entry in a BatchPlan against a shared per-layer K/V pool,
indexed by physical block id.
"""

import math

import torch
import triton
import triton.language as tl


# Grid: (N_Q // BLOCK_M, Z * H). Z=batches, H=heads, N_Q=query positions,
# N_K=key/value positions (cache + new).
@triton.jit
def flash_attention_2(
    Q, K, V, O,
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
    N_GROUPS: tl.constexpr,  # query heads per K/V head
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    off_z = off_hz // H
    off_h = off_hz % H

    q_offset = off_z * stride_qz + off_h * stride_qh
    off_kv_h = off_h // N_GROUPS
    k_offset = off_z * stride_kz + off_kv_h * stride_kh
    v_offset = off_z * stride_vz + off_kv_h * stride_vh

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
    """Q: (Z, H, N_Q, D); K, V: (Z, H_KV, N_K, D) with H a multiple of H_KV. Returns O,
    shaped like Q. N_Q and N_K may differ: a decode step passes N_Q=1 against the
    sequence's full cached N_K, with query_offset=N_K-N_Q so the causal mask compares true
    sequence positions.
    """
    Z, H, N_Q, D = q.shape
    Zk, H_KV, N_K, Dk = k.shape
    assert (Z, D) == (Zk, Dk), "Q and K/V must share batch and head_dim"
    assert H % H_KV == 0, f"query heads ({H}) must be a multiple of KV heads ({H_KV})"
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
    BLOCK_M, BLOCK_N = _TILES[q.dtype]
    BLOCK_D = triton.next_power_of_2(D)
    grid = (triton.cdiv(N_Q, BLOCK_M), Z * H)

    try:
        flash_attention_2[grid](
            q, k, v, o,
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
            N_GROUPS=H // H_KV,
        )
    except triton.runtime.errors.OutOfResources as e:
        raise _oom_hint(e, D, q.dtype, BLOCK_M, BLOCK_N) from None

    return o


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


# `paged_attention_decode_forward` is `paged_attention_forward`'s decode-only sibling: every
# entry's query is exactly one row (q_len 0 or 1 -- 0 only for a graph bucket's padding), so
# batch 1 no longer means "one program per head": a query offset's whole context splits across
# N_SPLITS programs per head, each an independent online-softmax pass over its slice, combined
# by a second kernel. At batch 1 that turns a 24-program launch (one per head, one query row
# each) into N_SPLITS times as many, on a GPU with far more SMs than heads.
@triton.jit
def paged_attention_decode_splitk(
    Q, K_POOL, V_POOL,
    PARTIAL_ACC, PARTIAL_M, PARTIAL_L,
    BLOCK_TABLE, CONTEXT_LEN, Q_START, Q_LEN,
    stride_qh, stride_qm, stride_qd,
    stride_kb, stride_ks, stride_kh, stride_kd,
    stride_vb, stride_vs, stride_vh, stride_vd,
    stride_pa_s, stride_pa_e, stride_pa_h,
    stride_pml_s, stride_pml_e, stride_pml_h,
    stride_bt_row,
    H, D,
    BLOCK_SIZE_KV,
    softmax_scale,
    N_GROUPS: tl.constexpr,
    N_SPLITS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """One program per (split, entry, query head): partial (acc, m, l) over context
    slice [split * ceil(context_len / N_SPLITS), ...) of that entry's causal context.
    A padding row (Q_LEN 0) or a split past its entry's context writes the neutral
    partial (m=-inf, l=0), which contributes nothing once combined."""
    split_id = tl.program_id(0)
    entry_head = tl.program_id(1)
    entry_idx = entry_head // H
    head_idx = entry_head % H
    kv_head_idx = head_idx // N_GROUPS

    pa_offset = split_id * stride_pa_s + entry_idx * stride_pa_e + head_idx * stride_pa_h
    pml_offset = split_id * stride_pml_s + entry_idx * stride_pml_e + head_idx * stride_pml_h
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < D

    q_len = tl.load(Q_LEN + entry_idx)
    if q_len == 0:
        # A padding row: combine returns on Q_LEN before reading any of its partials, so
        # this store only keeps every slot of the workspace uniformly written.
        tl.store(PARTIAL_ACC + pa_offset + offs_d, 0.0, mask=d_mask)
        tl.store(PARTIAL_M + pml_offset, float("-inf"))
        tl.store(PARTIAL_L + pml_offset, 0.0)
        return

    context_len = tl.load(CONTEXT_LEN + entry_idx)
    chunk = tl.cdiv(context_len, N_SPLITS)
    start_n = split_id * chunk
    end_n = tl.minimum(start_n + chunk, context_len)
    if start_n >= end_n:
        # A split past this real row's context: rescale = exp(-inf - global_max) = 0 in
        # the combine kernel, so this acc is never actually weighted in -- but combine
        # unconditionally reads it (0 * uninitialized memory is not always 0), so it must
        # hold a finite value, not whatever the allocator gave it.
        tl.store(PARTIAL_ACC + pa_offset + offs_d, 0.0, mask=d_mask)
        tl.store(PARTIAL_M + pml_offset, float("-inf"))
        tl.store(PARTIAL_L + pml_offset, 0.0)
        return

    q_start = tl.load(Q_START + entry_idx)
    q_ptrs = Q + head_idx * stride_qh + q_start * stride_qm + offs_d * stride_qd
    q = tl.load(q_ptrs, mask=d_mask, other=0.0).to(tl.float32)

    running_max = float("-inf")
    running_denom = 0.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    for start in range(0, BLOCK_N * tl.cdiv(chunk, BLOCK_N), BLOCK_N):
        offs_n = start_n + start + tl.arange(0, BLOCK_N)
        n_mask = offs_n < end_n

        block_in_table = offs_n // BLOCK_SIZE_KV
        within_block = offs_n % BLOCK_SIZE_KV
        block_id = tl.load(
            BLOCK_TABLE + entry_idx * stride_bt_row + block_in_table, mask=n_mask, other=0
        )

        k_ptrs = (K_POOL + block_id[:, None] * stride_kb + within_block[:, None] * stride_ks
                  + kv_head_idx * stride_kh + offs_d[None, :] * stride_kd)
        k = tl.load(k_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)

        # One query row: an elementwise product and a reduction, not tl.dot -- a single
        # row gets no tensor-core benefit and tl.dot needs a taller tile than M=1 gives it.
        s = tl.sum(q[None, :] * k.to(tl.float32), axis=1) * softmax_scale
        s = tl.where(n_mask, s, float("-inf"))

        tile_max = tl.max(s, axis=0)
        new_max = tl.maximum(running_max, tile_max)
        alpha = tl.exp(running_max - new_max)
        p = tl.exp(s - new_max)

        v_ptrs = (V_POOL + block_id[:, None] * stride_vb + within_block[:, None] * stride_vs
                  + kv_head_idx * stride_vh + offs_d[None, :] * stride_vd)
        v = tl.load(v_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)

        running_denom = alpha * running_denom + tl.sum(p, axis=0)
        acc = acc * alpha + tl.sum(p[:, None] * v.to(tl.float32), axis=0)
        running_max = new_max

    tl.store(PARTIAL_ACC + pa_offset + offs_d, acc, mask=d_mask)
    tl.store(PARTIAL_M + pml_offset, running_max)
    tl.store(PARTIAL_L + pml_offset, running_denom)


@triton.jit
def paged_attention_decode_combine(
    PARTIAL_ACC, PARTIAL_M, PARTIAL_L, O,
    Q_START, Q_LEN,
    stride_pa_s, stride_pa_e, stride_pa_h,
    stride_pml_s, stride_pml_e, stride_pml_h,
    stride_oh, stride_om, stride_od,
    H, D,
    N_SPLITS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """One program per (entry, query head): rescales each split's partial softmax onto
    their shared true max and sums them, the same combine step online softmax runs
    between tiles, just across kernel launches instead of within one."""
    entry_head = tl.program_id(0)
    entry_idx = entry_head // H
    head_idx = entry_head % H
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < D

    q_len = tl.load(Q_LEN + entry_idx)
    if q_len == 0:
        return  # padding row: its output is never read downstream, so there's nothing to write

    q_start = tl.load(Q_START + entry_idx)
    o_ptrs = O + head_idx * stride_oh + q_start * stride_om + offs_d * stride_od

    pml_base = entry_idx * stride_pml_e + head_idx * stride_pml_h
    global_max = float("-inf")
    for s in range(N_SPLITS):
        m = tl.load(PARTIAL_M + s * stride_pml_s + pml_base)
        global_max = tl.maximum(global_max, m)

    pa_base = entry_idx * stride_pa_e + head_idx * stride_pa_h
    numer = tl.zeros([BLOCK_D], dtype=tl.float32)
    denom = 0.0
    for s in range(N_SPLITS):
        m = tl.load(PARTIAL_M + s * stride_pml_s + pml_base)
        length = tl.load(PARTIAL_L + s * stride_pml_s + pml_base)
        rescale = tl.exp(m - global_max)
        acc = tl.load(PARTIAL_ACC + s * stride_pa_s + pa_base + offs_d, mask=d_mask, other=0.0)
        numer += rescale * acc
        denom += rescale * length

    tl.store(o_ptrs, numer / denom, mask=d_mask)


def paged_attention_decode_forward(
    q: torch.Tensor, k_pool: torch.Tensor, v_pool: torch.Tensor,
    block_table: torch.Tensor, context_len: torch.Tensor,
    q_start: torch.Tensor, q_len: torch.Tensor, block_size: int,
    n_splits: int,
    workspace: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Split-K paged decode attention, for a BatchPlan where every entry's query is one
    row: split each entry's context across `n_splits` programs per head, then combine
    their partial softmaxes. `workspace` is `(partial_acc, partial_m, partial_l)`,
    shaped for exactly this call's `q.shape[0]` (heads) and `block_table.shape[0]`
    (entries) -- pass a graph's own pre-allocated one when replaying this call inside a
    CUDA graph, so a captured decode iteration allocates nothing per layer or per
    replay. Omit it for an eager call, which allocates its own each time.

    q: (H, total_tokens, D). k_pool/v_pool: (num_blocks, block_size, H_KV, D).
    block_table/context_len/q_start/q_len: one row per batch entry.
    """
    H, _total_tokens, D = q.shape
    H_KV = k_pool.shape[2]
    num_entries = block_table.shape[0]
    assert q.is_cuda and k_pool.is_cuda and v_pool.is_cuda, "Q, K, V not all CUDA tensors"
    assert q.dtype == k_pool.dtype == v_pool.dtype, "Q, K, V must share a dtype"
    assert H % H_KV == 0, f"query heads ({H}) must be a multiple of KV heads ({H_KV})"

    BLOCK_D = triton.next_power_of_2(D)
    BLOCK_N = 64

    if workspace is None:
        partial_acc = torch.empty(
            (n_splits, num_entries, H, D), dtype=torch.float32, device=q.device
        )
        partial_m = torch.empty((n_splits, num_entries, H), dtype=torch.float32, device=q.device)
        partial_l = torch.empty((n_splits, num_entries, H), dtype=torch.float32, device=q.device)
    else:
        partial_acc, partial_m, partial_l = workspace

    o = torch.empty_like(q)

    paged_attention_decode_splitk[(n_splits, num_entries * H)](
        q, k_pool, v_pool,
        partial_acc, partial_m, partial_l,
        block_table, context_len, q_start, q_len,
        q.stride(0), q.stride(1), q.stride(2),
        k_pool.stride(0), k_pool.stride(1), k_pool.stride(2), k_pool.stride(3),
        v_pool.stride(0), v_pool.stride(1), v_pool.stride(2), v_pool.stride(3),
        partial_acc.stride(0), partial_acc.stride(1), partial_acc.stride(2),
        partial_m.stride(0), partial_m.stride(1), partial_m.stride(2),
        block_table.stride(0),
        H, D,
        block_size,
        1.0 / math.sqrt(D),
        N_GROUPS=H // H_KV,
        N_SPLITS=n_splits,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
    )
    paged_attention_decode_combine[(num_entries * H,)](
        partial_acc, partial_m, partial_l, o,
        q_start, q_len,
        partial_acc.stride(0), partial_acc.stride(1), partial_acc.stride(2),
        partial_m.stride(0), partial_m.stride(1), partial_m.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        H, D,
        N_SPLITS=n_splits,
        BLOCK_D=BLOCK_D,
    )
    return o


def decode_workspace(
    n_splits: int, num_entries: int, n_heads: int, head_dim: int, device: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A `paged_attention_decode_forward` workspace sized for one bucket, to allocate
    once before a CUDA graph captures replaying against it. `partial_acc`'s last
    dimension (head_dim) is addressed as offsets, not a stride, so it must stay
    contiguous -- true of a fresh `torch.empty` and never violated by a reshape or
    slice elsewhere in this module. `partial_m` and `partial_l` share one shape, so
    the kernels address both through `partial_m`'s strides."""
    return (
        torch.empty(
            (n_splits, num_entries, n_heads, head_dim), dtype=torch.float32, device=device
        ),
        torch.empty((n_splits, num_entries, n_heads), dtype=torch.float32, device=device),
        torch.empty((n_splits, num_entries, n_heads), dtype=torch.float32, device=device),
    )


# Programs a decode iteration aims for, batch * n_heads * n_splits: several times a
# modern GPU's SM count, so irregular split lengths still leave enough scheduling slack
# to keep every SM busy. Well past this many running sequences, the plain per-head grid
# (batch * n_heads programs, no splitting) already covers the GPU on its own.
DECODE_SPLIT_TARGET_PROGRAMS = 512


def decode_splits(batch_size: int) -> int:
    """Split count for a decode iteration of `batch_size` running sequences: enough that
    `batch_size * n_heads * n_splits` can fill the GPU's SMs, capped at 16 so a short
    context isn't cut into far more pieces than it has keys to spread across."""
    return max(1, min(DECODE_SPLIT_TARGET_PROGRAMS // max(batch_size, 1), 16))
