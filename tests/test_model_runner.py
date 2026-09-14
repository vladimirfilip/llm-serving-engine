from __future__ import annotations

import time

import pytest

pytest.importorskip("transformers")
torch = pytest.importorskip("torch")

MODEL = "sshleifer/tiny-gpt2"


@pytest.fixture(scope="module")
def loaded_runner():
    from llm_serving_engine.config import ModelConfig
    from llm_serving_engine.model_runner import ModelRunner

    try:
        runner = ModelRunner(
            ModelConfig(model_name_or_path=MODEL, device="cpu", use_custom_kernels=False)
        )
    except Exception as e:
        pytest.skip(f"can't fetch {MODEL} from the HF Hub: {e}")
    return runner


def _make_sequence(seq_id: int, prompt_tokens: list[int], **sampling_overrides):
    from llm_serving_engine.metrics import RequestMetrics
    from llm_serving_engine.sampling import SamplingParams
    from llm_serving_engine.sequence import Sequence

    return Sequence(
        seq_id=seq_id,
        prompt_tokens=prompt_tokens,
        sampling_params=SamplingParams(**sampling_overrides),
        arrival_time=time.monotonic(),
        metrics=RequestMetrics(enqueue_time=time.monotonic()),
    )


def test_load_default_path_runs_on_cpu(loaded_runner):
    assert loaded_runner.device == "cpu"
    assert loaded_runner.model is not None


def _admit(seq, n_tokens):
    """Mirrors scheduler_step's admission bookkeeping for a single BatchEntry.

    is_prefill_chunk marks prefill work (vs. decode), not "partial chunk" — a one-shot
    admission that finishes the whole prompt is still prefill, per scheduler.py.
    """
    from llm_serving_engine.batch_plan import BatchEntry

    seq.prefill_progress += n_tokens
    seq.status = "PREFILLING" if n_tokens < len(seq.prompt_tokens) else "DECODING"
    return BatchEntry(seq.seq_id, n_tokens, is_prefill_chunk=True)


def test_forward_one_shot_prefill_returns_token_and_primes_cache(loaded_runner):
    from llm_serving_engine.batch_plan import BatchPlan

    seq = _make_sequence(1, [1, 2, 3, 4], temperature=0.0, max_tokens=8)
    plan = BatchPlan()
    plan.entries.append(_admit(seq, len(seq.prompt_tokens)))

    results = loaded_runner.forward(plan, {seq.seq_id: seq})
    assert len(results) == 1
    seq_id, token_id, _finished = results[0]
    assert seq_id == seq.seq_id
    assert isinstance(token_id, int)
    assert seq.seq_id in loaded_runner._past_key_values
    loaded_runner.free(seq.seq_id)


def test_forward_partial_prefill_chunk_samples_nothing(loaded_runner):
    from llm_serving_engine.batch_plan import BatchPlan

    seq = _make_sequence(2, [1, 2, 3, 4], temperature=0.0, max_tokens=8)
    plan = BatchPlan()
    plan.entries.append(_admit(seq, 2))  # chunk shorter than the prompt

    results = loaded_runner.forward(plan, {seq.seq_id: seq})
    assert results == []  # cache extended, but nothing to sample yet
    assert seq.seq_id in loaded_runner._past_key_values
    loaded_runner.free(seq.seq_id)


def test_forward_decode_step_advances_and_reports_shape(loaded_runner):
    from llm_serving_engine.batch_plan import BatchEntry, BatchPlan

    seq = _make_sequence(3, [5, 6, 7], temperature=0.0, max_tokens=3)
    prefill_plan = BatchPlan()
    prefill_plan.entries.append(_admit(seq, len(seq.prompt_tokens)))
    first = loaded_runner.forward(prefill_plan, {seq.seq_id: seq})[0][1]
    seq.generated_tokens.append(first)

    decode_plan = BatchPlan()
    decode_plan.entries.append(BatchEntry(seq.seq_id, 1, is_prefill_chunk=False))
    results = loaded_runner.forward(decode_plan, {seq.seq_id: seq})
    assert len(results) == 1
    seq_id, token_id, _finished = results[0]
    assert seq_id == seq.seq_id
    assert isinstance(token_id, int)
    assert isinstance(_finished, bool)
    loaded_runner.free(seq.seq_id)


def test_forward_decode_step_finishes_at_max_tokens(loaded_runner):
    from llm_serving_engine.batch_plan import BatchEntry, BatchPlan

    seq = _make_sequence(4, [1, 2], temperature=0.0, max_tokens=2)
    prefill_plan = BatchPlan()
    prefill_plan.entries.append(_admit(seq, len(seq.prompt_tokens)))
    first = loaded_runner.forward(prefill_plan, {seq.seq_id: seq})[0][1]
    seq.generated_tokens.append(first)  # 1 of 2 tokens generated

    decode_plan = BatchPlan()
    decode_plan.entries.append(BatchEntry(seq.seq_id, 1, is_prefill_chunk=False))
    _, _, finished = loaded_runner.forward(decode_plan, {seq.seq_id: seq})[0]
    assert finished  # second token hits max_tokens
    loaded_runner.free(seq.seq_id)


def test_forward_decode_step_finishes_on_any_listed_eos_token(loaded_runner, monkeypatch):
    # Instruct models (Llama-3: <|eot_id|> alongside <|end_of_text|>) list multiple stop
    # tokens; finishing must check all of them, not just the first.
    from llm_serving_engine.batch_plan import BatchEntry, BatchPlan

    monkeypatch.setattr(loaded_runner.model.generation_config, "eos_token_id", [999, 111])
    seq = _make_sequence(5, [1, 2], temperature=0.0, max_tokens=50)
    prefill_plan = BatchPlan()
    prefill_plan.entries.append(_admit(seq, len(seq.prompt_tokens)))
    first = loaded_runner.forward(prefill_plan, {seq.seq_id: seq})[0][1]
    seq.generated_tokens.append(first)

    monkeypatch.setattr(loaded_runner, "_sample", lambda logits, params: 111)
    decode_plan = BatchPlan()
    decode_plan.entries.append(BatchEntry(seq.seq_id, 1, is_prefill_chunk=False))
    _, token_id, finished = loaded_runner.forward(decode_plan, {seq.seq_id: seq})[0]
    assert token_id == 111
    assert finished
    loaded_runner.free(seq.seq_id)


def test_free_drops_kv_cache(loaded_runner):
    from llm_serving_engine.batch_plan import BatchPlan

    seq = _make_sequence(6, [1, 2], temperature=0.0, max_tokens=4)
    plan = BatchPlan()
    plan.entries.append(_admit(seq, len(seq.prompt_tokens)))
    loaded_runner.forward(plan, {seq.seq_id: seq})
    loaded_runner.free(seq.seq_id)
    assert seq.seq_id not in loaded_runner._past_key_values


def test_quantize_int8_is_not_implemented():
    from llm_serving_engine.config import ModelConfig
    from llm_serving_engine.model_runner import ModelRunner

    with pytest.raises(NotImplementedError):
        ModelRunner(ModelConfig(model_name_or_path=MODEL, device="cpu", quantize="int8"))


def test_wire_custom_kernels_rejects_a_non_cuda_device(loaded_runner):
    with pytest.raises(RuntimeError, match="CUDA"):
        loaded_runner.wire_custom_kernels()


def test_wire_custom_kernels_rejects_a_non_llama_family_model():
    if not torch.cuda.is_available():
        pytest.skip("only device is checked without CUDA; this test needs the shape check")
    from transformers import GPT2Config, GPT2LMHeadModel

    from llm_serving_engine.config import ModelConfig
    from llm_serving_engine.model_runner import ModelRunner

    runner = ModelRunner.__new__(ModelRunner)
    runner.config = ModelConfig(model_name_or_path="unused", device="cuda", use_custom_kernels=True)
    runner.device = "cuda"
    runner.model = GPT2LMHeadModel(GPT2Config(n_layer=1, n_head=1, n_embd=8)).to("cuda").eval()

    with pytest.raises(RuntimeError, match="Llama-family"):
        runner.wire_custom_kernels()


def test_wire_custom_kernels_rejects_fp16_with_a_head_dim_under_16():
    if not torch.cuda.is_available():
        pytest.skip("only device is checked without CUDA; this test needs the dtype/head_dim check")
    from transformers import LlamaConfig, LlamaForCausalLM

    from llm_serving_engine.config import ModelConfig
    from llm_serving_engine.model_runner import ModelRunner

    # head_dim = hidden_size / num_attention_heads = 16 / 4 = 4: the fp16 tensor-core
    # dot product the Triton kernel issues for Q@K^T needs a contraction dim >= 16.
    hf_config = LlamaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=32,
    )
    runner = ModelRunner.__new__(ModelRunner)
    runner.config = ModelConfig(model_name_or_path="unused", device="cuda", use_custom_kernels=True)
    runner.device = "cuda"
    runner.model = LlamaForCausalLM(hf_config).to("cuda").to(torch.float16).eval()

    with pytest.raises(RuntimeError, match="head_dim"):
        runner.wire_custom_kernels()


def test_flatten_plan_offsets_and_positions(loaded_runner):
    from llm_serving_engine.batch_plan import BatchEntry, BatchPlan

    prefilling = _make_sequence(20, [1, 2, 3, 4], temperature=0.0, max_tokens=8)
    prefilling.prefill_progress = 3
    prefilling.status = "PREFILLING"

    decoding = _make_sequence(21, [9], temperature=0.0, max_tokens=8)
    decoding.generated_tokens.append(42)

    plan = BatchPlan()
    plan.entries.append(BatchEntry(prefilling.seq_id, 3, is_prefill_chunk=True))
    plan.entries.append(BatchEntry(decoding.seq_id, 1, is_prefill_chunk=False))

    input_ids, position_ids, offsets = loaded_runner._flatten_plan(
        plan, {prefilling.seq_id: prefilling, decoding.seq_id: decoding}
    )
    assert input_ids.tolist() == [[1, 2, 3, 42]]
    assert position_ids.tolist() == [[0, 1, 2, 1]]
    assert offsets == [(0, 3), (3, 4)]


def _build_tiny_llama_runner():
    """A random tiny LlamaConfig (GQA: 4 heads / 2 kv heads) needs no network or pretrained
    weights — it's the smallest thing shaped like the Llama-family decoder attention_forward
    assumes, letting these tests exercise the real CUDA-only Triton kernels end to end."""
    from transformers import LlamaConfig, LlamaForCausalLM

    from llm_serving_engine.config import ModelConfig
    from llm_serving_engine.model_runner import ModelRunner

    hf_config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    runner = ModelRunner.__new__(ModelRunner)
    runner.config = ModelConfig(model_name_or_path="unused", device="cuda", use_custom_kernels=True)
    runner.device = "cuda"
    runner._past_key_values = {}
    runner._kv_cache = {}
    runner._k_pool = None
    runner._v_pool = None
    runner._block_size = None
    runner._scratch_block_id = None
    runner.model = LlamaForCausalLM(hf_config).to("cuda").eval()
    from llm_serving_engine.decode_graph import DecodeGraphRunner

    runner._decode_graphs = DecodeGraphRunner(runner)
    return runner


@pytest.fixture(scope="module")
def tiny_llama_runner():
    if not torch.cuda.is_available():
        pytest.skip("attention_forward calls a CUDA-only Triton kernel")
    pytest.importorskip("triton")
    return _build_tiny_llama_runner()


@pytest.fixture(scope="module")
def paged_llama_runner():
    """A separate instance from tiny_llama_runner: allocate_kv_pool mutates the runner
    (attention_forward's paged path is picked up automatically once a pool exists), and
    the non-paged tests above rely on sharing one fixture with no pool ever allocated."""
    if not torch.cuda.is_available():
        pytest.skip("attention_forward calls a CUDA-only Triton kernel")
    pytest.importorskip("triton")
    runner = _build_tiny_llama_runner()
    runner.allocate_kv_pool(num_blocks=64, block_size=4)
    return runner


@pytest.fixture(scope="module")
def graphed_llama_runner():
    """A third, separate instance: capture_decode_graphs captures against this specific
    model's weights and pool, so it can't be shared with paged_llama_runner (whose tests
    exercise the non-graphed paged path and shouldn't silently start taking the graphed
    one) or across test runs within this module (each capture is tied to this process's
    CUDA context)."""
    if not torch.cuda.is_available():
        pytest.skip("attention_forward calls a CUDA-only Triton kernel")
    pytest.importorskip("triton")
    runner = _build_tiny_llama_runner()
    runner.allocate_kv_pool(num_blocks=64, block_size=4)
    runner.capture_decode_graphs(bucket_sizes=[1, 2, 4, 8])
    assert runner._decode_graphs, "capture/self-check failed in test setup"
    return runner


def _admit_seq(allocator, seq_id, prompt_len):
    """Builds and fully admits (via a real BlockAllocator) a fresh decoding sequence, used
    by the graphed-decode tests below where each sequence's real block ids matter (unlike
    attention_forward's own tests, which mostly use hand-built BatchPlans against a
    shared fixture)."""
    seq = _make_sequence(seq_id, list(range(3, 3 + prompt_len)), temperature=0.0, max_tokens=32)
    assert allocator.get_capacity(seq, prompt_len)
    return seq


def _normed_layer0(runner, plan, seqs):
    input_ids, position_ids, offsets = runner._flatten_plan(plan, seqs)
    with torch.no_grad():
        hidden = runner.model.get_input_embeddings()(input_ids)
        position_embeddings = runner.model.model.rotary_emb(hidden, position_ids)
        layer0 = runner._layers[0]
        normed = layer0.input_layernorm(hidden)
    return layer0, normed, position_embeddings, offsets


def test_attention_forward_full_prefill_matches_hf_reference(tiny_llama_runner):
    from llm_serving_engine.batch_plan import BatchEntry, BatchPlan

    seq = _make_sequence(30, [1, 2, 3, 4, 5], temperature=0.0, max_tokens=8)
    seq.prefill_progress = 5
    seq.status = "DECODING"
    plan = BatchPlan()
    plan.entries.append(BatchEntry(seq.seq_id, 5, is_prefill_chunk=True))
    seqs = {seq.seq_id: seq}

    layer0, normed, position_embeddings, offsets = _normed_layer0(tiny_llama_runner, plan, seqs)
    with torch.no_grad():
        ref_out, _ = layer0.self_attn(normed, position_embeddings=position_embeddings)
        mine = tiny_llama_runner.attention_forward(
            normed, plan, seqs, offsets, position_embeddings, layer_idx=0
        )

    assert torch.allclose(ref_out, mine, atol=1e-3)


def _hf_full_sequence_reference(runner, layer0, token_ids: list[int]):
    """Ground truth for a whole sequence in one non-incremental call: eager attention with
    an explicit causal mask (HF's default of no mask when attention_mask=None isn't causal,
    so this can't rely on that default the way a plain prefill call can).
    """
    tokens = torch.tensor([token_ids], device=runner.device)
    n = len(token_ids)
    with torch.no_grad():
        hidden = runner.model.get_input_embeddings()(tokens)
        position_ids = torch.arange(n, device=runner.device).unsqueeze(0)
        position_embeddings = runner.model.model.rotary_emb(hidden, position_ids)
        normed = layer0.input_layernorm(hidden)
        causal_mask = torch.full((n, n), float("-inf"), device=runner.device).triu(1)
        ref_out, _ = layer0.self_attn(
            normed, position_embeddings=position_embeddings, attention_mask=causal_mask
        )
    return ref_out


def test_attention_forward_decode_step_matches_hf_reference(tiny_llama_runner):
    """query_offset lets a decode step's single new token (Q_len=1) attend to its whole
    cache (K_len=6) — the shape the unmodified upstream kernel can't express at all.
    """
    from llm_serving_engine.batch_plan import BatchEntry, BatchPlan

    seq = _make_sequence(31, [3, 1, 4, 1, 5], temperature=0.0, max_tokens=8)
    seq.prefill_progress = 5
    seq.status = "DECODING"
    prefill_plan = BatchPlan()
    prefill_plan.entries.append(BatchEntry(seq.seq_id, 5, is_prefill_chunk=True))
    seqs = {seq.seq_id: seq}
    layer0, normed, position_embeddings, offsets = _normed_layer0(tiny_llama_runner, prefill_plan, seqs)
    with torch.no_grad():
        tiny_llama_runner.attention_forward(
            normed, prefill_plan, seqs, offsets, position_embeddings, layer_idx=0
        )

    seq.generated_tokens.append(9)
    decode_plan = BatchPlan()
    decode_plan.entries.append(BatchEntry(seq.seq_id, 1, is_prefill_chunk=False))
    _layer0, normed, position_embeddings, offsets = _normed_layer0(tiny_llama_runner, decode_plan, seqs)
    with torch.no_grad():
        mine = tiny_llama_runner.attention_forward(
            normed, decode_plan, seqs, offsets, position_embeddings, layer_idx=0
        )

    ref_out = _hf_full_sequence_reference(tiny_llama_runner, layer0, [3, 1, 4, 1, 5, 9])
    assert torch.allclose(ref_out[:, -1:], mine, atol=1e-3)


def test_attention_forward_chunked_prefill_continuation_matches_hf_reference(tiny_llama_runner):
    """A prefill chunk after the first also has query_offset > 0 (3 tokens already cached,
    3 new ones) — same causal-mask fix as decode, exercised with n_tokens > 1.
    """
    from llm_serving_engine.batch_plan import BatchEntry, BatchPlan

    seq = _make_sequence(32, [1, 2, 3, 4, 5, 6], temperature=0.0, max_tokens=8)
    seq.prefill_progress = 3
    seq.status = "PREFILLING"
    chunk1 = BatchPlan()
    chunk1.entries.append(BatchEntry(seq.seq_id, 3, is_prefill_chunk=True))
    seqs = {seq.seq_id: seq}
    layer0, normed, position_embeddings, offsets = _normed_layer0(tiny_llama_runner, chunk1, seqs)
    with torch.no_grad():
        tiny_llama_runner.attention_forward(
            normed, chunk1, seqs, offsets, position_embeddings, layer_idx=0
        )

    seq.prefill_progress = 6
    seq.status = "DECODING"
    chunk2 = BatchPlan()
    chunk2.entries.append(BatchEntry(seq.seq_id, 3, is_prefill_chunk=True))
    _layer0, normed, position_embeddings, offsets = _normed_layer0(tiny_llama_runner, chunk2, seqs)
    with torch.no_grad():
        mine = tiny_llama_runner.attention_forward(
            normed, chunk2, seqs, offsets, position_embeddings, layer_idx=0
        )

    ref_out = _hf_full_sequence_reference(tiny_llama_runner, layer0, [1, 2, 3, 4, 5, 6])
    assert torch.allclose(ref_out[:, 3:6], mine, atol=1e-3)


def test_attention_forward_disjoint_caches_stay_independent(tiny_llama_runner):
    """Two sequences interleaved through the same layer must not see each other's tokens —
    the whole point of a disjoint per-sequence cache instead of one shared buffer.
    """
    from llm_serving_engine.batch_plan import BatchEntry, BatchPlan

    seq_a = _make_sequence(33, [1, 1], temperature=0.0, max_tokens=8)
    seq_a.prefill_progress = 2
    seq_a.status = "DECODING"
    seq_b = _make_sequence(34, [2, 2, 2, 2], temperature=0.0, max_tokens=8)
    seq_b.prefill_progress = 4
    seq_b.status = "DECODING"
    seqs = {seq_a.seq_id: seq_a, seq_b.seq_id: seq_b}

    plan = BatchPlan()
    plan.entries.append(BatchEntry(seq_a.seq_id, 2, is_prefill_chunk=True))
    plan.entries.append(BatchEntry(seq_b.seq_id, 4, is_prefill_chunk=True))
    _layer0, normed, position_embeddings, offsets = _normed_layer0(tiny_llama_runner, plan, seqs)
    with torch.no_grad():
        tiny_llama_runner.attention_forward(
            normed, plan, seqs, offsets, position_embeddings, layer_idx=0
        )

    cache = tiny_llama_runner._kv_cache[0]
    assert cache[seq_a.seq_id][2] == 2  # filled length, not the preallocated capacity
    assert cache[seq_b.seq_id][2] == 4


def test_wire_custom_kernels_accepts_a_cuda_llama_model(tiny_llama_runner):
    tiny_llama_runner.wire_custom_kernels()  # must not raise


def test_forward_fused_prefill_then_decode_matches_hf_reference(tiny_llama_runner):
    """End-to-end forward_fused (embed, every layer's attention_forward + ffn_forward,
    final norm, lm head, sampling) against a plain non-incremental HF call on the same
    tokens — the two must agree since they compute the same function.
    """
    from llm_serving_engine.batch_plan import BatchEntry, BatchPlan

    prompt = [3, 1, 4, 1, 5]
    seq = _make_sequence(40, prompt, temperature=0.0, max_tokens=8)
    prefill_plan = BatchPlan()
    prefill_plan.entries.append(_admit(seq, len(prompt)))
    seqs = {seq.seq_id: seq}

    (result,) = tiny_llama_runner.forward_fused(prefill_plan, seqs)
    seq_id, token_id, finished = result
    assert seq_id == seq.seq_id
    assert not finished

    with torch.no_grad():
        ref_logits = tiny_llama_runner.model(torch.tensor([prompt], device="cuda")).logits
    assert token_id == int(torch.argmax(ref_logits[0, -1]).item())

    seq.generated_tokens.append(token_id)
    decode_plan = BatchPlan()
    decode_plan.entries.append(BatchEntry(seq.seq_id, 1, is_prefill_chunk=False))
    (result,) = tiny_llama_runner.forward_fused(decode_plan, seqs)
    _seq_id, token_id2, _finished = result

    with torch.no_grad():
        ref_logits = tiny_llama_runner.model(torch.tensor([[*prompt, token_id]], device="cuda")).logits
    assert token_id2 == int(torch.argmax(ref_logits[0, -1]).item())


def test_forward_fused_paged_prefill_then_decode_matches_hf_reference(paged_llama_runner):
    """Same contract as test_forward_fused_prefill_then_decode_matches_hf_reference, but
    routed through allocate_kv_pool's shared block-addressed pool (one Triton launch per
    layer covering the whole plan) instead of one disjoint buffer and launch per entry."""
    from llm_serving_engine.allocator import BlockAllocator
    from llm_serving_engine.batch_plan import BatchEntry, BatchPlan

    allocator = BlockAllocator(num_blocks=64, block_size=4)
    prompt = [3, 1, 4, 1, 5]
    seq = _make_sequence(50, prompt, temperature=0.0, max_tokens=8)
    seqs = {seq.seq_id: seq}

    assert allocator.get_capacity(seq, len(prompt))
    prefill_plan = BatchPlan()
    prefill_plan.entries.append(_admit(seq, len(prompt)))

    (result,) = paged_llama_runner.forward_fused(prefill_plan, seqs)
    seq_id, token_id, finished = result
    assert seq_id == seq.seq_id
    assert not finished

    with torch.no_grad():
        ref_logits = paged_llama_runner.model(torch.tensor([prompt], device="cuda")).logits
    assert token_id == int(torch.argmax(ref_logits[0, -1]).item())

    seq.generated_tokens.append(token_id)
    assert allocator.get_capacity(seq, 1)
    decode_plan = BatchPlan()
    decode_plan.entries.append(BatchEntry(seq.seq_id, 1, is_prefill_chunk=False))
    (result,) = paged_llama_runner.forward_fused(decode_plan, seqs)
    _seq_id, token_id2, _finished = result

    with torch.no_grad():
        ref_logits = paged_llama_runner.model(torch.tensor([[*prompt, token_id]], device="cuda")).logits
    assert token_id2 == int(torch.argmax(ref_logits[0, -1]).item())

    allocator.free(seq.block_table)


def test_forward_fused_paged_disjoint_sequences_match_hf_reference(paged_llama_runner):
    """Two different-length sequences sharing one pool, admitted through a real
    BlockAllocator so their physical blocks differ, must not see each other's tokens --
    checked against each one's own non-incremental HF reference, not just by comparing
    block ids, so a gather-addressing bug would show up as a wrong token, not just a
    suspicious-looking block table.
    """
    from llm_serving_engine.allocator import BlockAllocator
    from llm_serving_engine.batch_plan import BatchPlan

    allocator = BlockAllocator(num_blocks=64, block_size=4)
    prompt_a, prompt_b = [1, 1, 3], [2, 2, 2, 2, 2]
    seq_a = _make_sequence(51, prompt_a, temperature=0.0, max_tokens=8)
    seq_b = _make_sequence(52, prompt_b, temperature=0.0, max_tokens=8)
    seqs = {seq_a.seq_id: seq_a, seq_b.seq_id: seq_b}

    assert allocator.get_capacity(seq_a, len(prompt_a))
    assert allocator.get_capacity(seq_b, len(prompt_b))
    assert set(seq_a.block_table.physical_blocks).isdisjoint(seq_b.block_table.physical_blocks)

    plan = BatchPlan()
    plan.entries.append(_admit(seq_a, len(prompt_a)))
    plan.entries.append(_admit(seq_b, len(prompt_b)))

    result_a, result_b = paged_llama_runner.forward_fused(plan, seqs)

    with torch.no_grad():
        ref_a = paged_llama_runner.model(torch.tensor([prompt_a], device="cuda")).logits
        ref_b = paged_llama_runner.model(torch.tensor([prompt_b], device="cuda")).logits
    assert result_a[1] == int(torch.argmax(ref_a[0, -1]).item())
    assert result_b[1] == int(torch.argmax(ref_b[0, -1]).item())

    allocator.free(seq_a.block_table)
    allocator.free(seq_b.block_table)


def test_forward_fused_paged_mixed_decode_and_prefill_in_one_plan(paged_llama_runner):
    """A decode entry (q_len=1, deep into its cache) and a fresh prefill entry (q_len>1,
    empty cache) in the same BatchPlan and the same grid launch -- the shape every real
    continuous-batching iteration takes once more than one sequence is in flight.
    """
    from llm_serving_engine.allocator import BlockAllocator
    from llm_serving_engine.batch_plan import BatchEntry, BatchPlan

    allocator = BlockAllocator(num_blocks=64, block_size=4)
    prompt_a = [4, 4, 4]
    seq_a = _make_sequence(53, prompt_a, temperature=0.0, max_tokens=8)
    seqs = {seq_a.seq_id: seq_a}
    assert allocator.get_capacity(seq_a, len(prompt_a))
    prefill_plan = BatchPlan()
    prefill_plan.entries.append(_admit(seq_a, len(prompt_a)))
    (result,) = paged_llama_runner.forward_fused(prefill_plan, seqs)
    seq_a.generated_tokens.append(result[1])
    assert allocator.get_capacity(seq_a, 1)

    prompt_b = [6, 6, 6, 6]
    seq_b = _make_sequence(54, prompt_b, temperature=0.0, max_tokens=8)
    seqs[seq_b.seq_id] = seq_b
    assert allocator.get_capacity(seq_b, len(prompt_b))

    plan = BatchPlan()
    plan.entries.append(BatchEntry(seq_a.seq_id, 1, is_prefill_chunk=False))
    plan.entries.append(_admit(seq_b, len(prompt_b)))

    result_a, result_b = paged_llama_runner.forward_fused(plan, seqs)

    with torch.no_grad():
        full_a = [*prompt_a, seq_a.generated_tokens[0]]
        ref_a = paged_llama_runner.model(torch.tensor([full_a], device="cuda")).logits
        ref_b = paged_llama_runner.model(torch.tensor([prompt_b], device="cuda")).logits
    assert result_a[1] == int(torch.argmax(ref_a[0, -1]).item())
    assert result_b[1] == int(torch.argmax(ref_b[0, -1]).item())

    allocator.free(seq_a.block_table)
    allocator.free(seq_b.block_table)


def test_allocate_kv_pool_shape(paged_llama_runner):
    cfg = paged_llama_runner.model.config
    assert len(paged_llama_runner._k_pool) == cfg.num_hidden_layers
    # +1: allocate_kv_pool reserves one extra row (block id == num_blocks) that
    # BlockAllocator never hands out, as a scratch target for graphed decode padding.
    expected = (65, 4, cfg.num_key_value_heads, cfg.hidden_size // cfg.num_attention_heads)
    assert tuple(paged_llama_runner._k_pool[0].shape) == expected
    assert tuple(paged_llama_runner._v_pool[0].shape) == expected
    assert paged_llama_runner._scratch_block_id == 64


def _prefill_to_decoding(runner, allocator, seq):
    """Runs one full (unchunked) prefill through forward_fused and leaves `seq` ready for
    a decode step: generated_tokens has its first token, and the block table already has
    capacity reserved for the next one -- exactly the state scheduler_step would leave a
    freshly-admitted sequence in by the time it first appears in a decode-only plan."""
    from llm_serving_engine.batch_plan import BatchPlan

    prefill_plan = BatchPlan()
    prefill_plan.entries.append(_admit(seq, len(seq.prompt_tokens)))
    (result,) = runner.forward_fused(prefill_plan, {seq.seq_id: seq})
    seq.generated_tokens.append(result[1])
    assert allocator.get_capacity(seq, 1)


def test_forward_graphed_matches_eager_no_padding(graphed_llama_runner):
    """Decode batch size exactly equal to a bucket (no padding rows at all)."""
    import copy

    from llm_serving_engine.allocator import BlockAllocator
    from llm_serving_engine.batch_plan import BatchEntry, BatchPlan

    allocator = BlockAllocator(num_blocks=64, block_size=4)
    seqs = {}
    for i, plen in enumerate([3, 5, 2, 4]):  # bucket=4, no padding
        seq = _admit_seq(allocator, 400 + i, plen)
        _prefill_to_decoding(graphed_llama_runner, allocator, seq)
        seqs[seq.seq_id] = seq

    decode_plan = BatchPlan()
    for seq_id in seqs:
        decode_plan.entries.append(BatchEntry(seq_id, 1, is_prefill_chunk=False))

    eager_seqs = copy.deepcopy(seqs)
    eager = graphed_llama_runner.forward_fused(decode_plan, eager_seqs)
    graphed = graphed_llama_runner.forward(decode_plan, seqs)
    assert sorted(graphed) == sorted(eager)

    for seq in seqs.values():
        allocator.free(seq.block_table)


def test_forward_graphed_matches_eager_with_padding(graphed_llama_runner):
    """3 real decode entries rounded up to bucket=4 -- exercises one padding row."""
    import copy

    from llm_serving_engine.allocator import BlockAllocator
    from llm_serving_engine.batch_plan import BatchEntry, BatchPlan

    allocator = BlockAllocator(num_blocks=64, block_size=4)
    seqs = {}
    for i, plen in enumerate([6, 2, 9]):
        seq = _admit_seq(allocator, 410 + i, plen)
        _prefill_to_decoding(graphed_llama_runner, allocator, seq)
        seqs[seq.seq_id] = seq

    decode_plan = BatchPlan()
    for seq_id in seqs:
        decode_plan.entries.append(BatchEntry(seq_id, 1, is_prefill_chunk=False))

    eager_seqs = copy.deepcopy(seqs)
    eager = graphed_llama_runner.forward_fused(decode_plan, eager_seqs)
    graphed = graphed_llama_runner.forward(decode_plan, seqs)
    assert sorted(graphed) == sorted(eager)

    for seq in seqs.values():
        allocator.free(seq.block_table)


def test_forward_graphed_padding_does_not_touch_other_pool_blocks(graphed_llama_runner):
    """A heavily-padded graphed decode (1 real row in a bucket=8 graph, 7 padding rows)
    must never write outside the reserved scratch block -- directly checks the bug the
    padding scheme exists to prevent: an unconditional pool write landing on a block a
    live, unrelated sequence owns.
    """
    from llm_serving_engine.allocator import BlockAllocator
    from llm_serving_engine.batch_plan import BatchEntry, BatchPlan

    allocator = BlockAllocator(num_blocks=64, block_size=4)
    seq = _admit_seq(allocator, 420, 3)
    _prefill_to_decoding(graphed_llama_runner, allocator, seq)
    seqs = {seq.seq_id: seq}

    sentinel_block = 50  # not the scratch block (64), not owned by `seq`
    sentinel = 12345.0
    for layer_idx in range(len(graphed_llama_runner._k_pool)):
        graphed_llama_runner._k_pool[layer_idx][sentinel_block].fill_(sentinel)
        graphed_llama_runner._v_pool[layer_idx][sentinel_block].fill_(sentinel)

    decode_plan = BatchPlan()
    decode_plan.entries.append(BatchEntry(seq.seq_id, 1, is_prefill_chunk=False))
    graphed_llama_runner.forward(decode_plan, seqs)  # 1 real row, bucket=8 -> 7 padding rows

    for layer_idx in range(len(graphed_llama_runner._k_pool)):
        assert torch.all(graphed_llama_runner._k_pool[layer_idx][sentinel_block] == sentinel)
        assert torch.all(graphed_llama_runner._v_pool[layer_idx][sentinel_block] == sentinel)

    allocator.free(seq.block_table)


def test_forward_graphed_row_reassignment_matches_eager(graphed_llama_runner):
    """Two consecutive graphed iterations where the occupied rows change (one sequence
    finishes-equivalent scope ends, a different one is admitted) -- catches both a stale
    q_len reactivating a now-padding row and a stale dest_block_id leaking a write into a
    block that row no longer owns.
    """
    import copy

    from llm_serving_engine.allocator import BlockAllocator
    from llm_serving_engine.batch_plan import BatchEntry, BatchPlan

    allocator = BlockAllocator(num_blocks=64, block_size=4)
    seqs = {}
    for i, plen in enumerate([4, 7]):
        seq = _admit_seq(allocator, 430 + i, plen)
        _prefill_to_decoding(graphed_llama_runner, allocator, seq)
        seqs[seq.seq_id] = seq

    decode_plan = BatchPlan()
    for seq_id in seqs:
        decode_plan.entries.append(BatchEntry(seq_id, 1, is_prefill_chunk=False))
    results = graphed_llama_runner.forward(decode_plan, seqs)
    for seq_id, token, _finished in results:
        seqs[seq_id].generated_tokens.append(token)
        assert allocator.get_capacity(seqs[seq_id], 1)

    # Iteration 2: admit a third sequence, decode all three together -- different rows,
    # different real batch size, same bucket (still rounds up to 4).
    new_seq = _admit_seq(allocator, 432, 5)
    _prefill_to_decoding(graphed_llama_runner, allocator, new_seq)
    seqs[new_seq.seq_id] = new_seq

    decode_plan2 = BatchPlan()
    for seq_id in seqs:
        decode_plan2.entries.append(BatchEntry(seq_id, 1, is_prefill_chunk=False))

    eager_seqs = copy.deepcopy(seqs)
    eager2 = graphed_llama_runner.forward_fused(decode_plan2, eager_seqs)
    graphed2 = graphed_llama_runner.forward(decode_plan2, seqs)
    assert sorted(graphed2) == sorted(eager2)

    for seq in seqs.values():
        allocator.free(seq.block_table)


def test_forward_graphed_replay_reflects_new_inputs(graphed_llama_runner):
    """Two different real plans at the same bucket must produce different logits --
    catches a forgotten .copy_() before replay, which would otherwise silently keep
    returning whatever was captured the first time."""
    from llm_serving_engine.allocator import BlockAllocator
    from llm_serving_engine.batch_plan import BatchEntry, BatchPlan

    allocator = BlockAllocator(num_blocks=64, block_size=4)
    seq_a = _admit_seq(allocator, 440, 3)
    _prefill_to_decoding(graphed_llama_runner, allocator, seq_a)
    seq_b = _admit_seq(allocator, 441, 11)
    _prefill_to_decoding(graphed_llama_runner, allocator, seq_b)

    plan_a = BatchPlan()
    plan_a.entries.append(BatchEntry(seq_a.seq_id, 1, is_prefill_chunk=False))
    graphed_llama_runner.forward(plan_a, {seq_a.seq_id: seq_a})
    bucket = min(b for b in graphed_llama_runner._decode_graphs if b >= 1)
    logits_a = graphed_llama_runner._decode_graphs[bucket].buffers.logits.clone()

    plan_b = BatchPlan()
    plan_b.entries.append(BatchEntry(seq_b.seq_id, 1, is_prefill_chunk=False))
    graphed_llama_runner.forward(plan_b, {seq_b.seq_id: seq_b})
    logits_b = graphed_llama_runner._decode_graphs[bucket].buffers.logits.clone()

    assert not torch.allclose(logits_a, logits_b), "replay returned stale, capture-time output"

    allocator.free(seq_a.block_table)
    allocator.free(seq_b.block_table)


def test_forward_dispatch_skips_graphs_for_mixed_prefill_and_decode(graphed_llama_runner, monkeypatch):
    from llm_serving_engine.allocator import BlockAllocator
    from llm_serving_engine.batch_plan import BatchEntry, BatchPlan

    allocator = BlockAllocator(num_blocks=64, block_size=4)
    decoding_seq = _admit_seq(allocator, 450, 3)
    _prefill_to_decoding(graphed_llama_runner, allocator, decoding_seq)
    prefill_seq = _admit_seq(allocator, 451, 5)
    seqs = {decoding_seq.seq_id: decoding_seq, prefill_seq.seq_id: prefill_seq}

    plan = BatchPlan()
    plan.entries.append(BatchEntry(decoding_seq.seq_id, 1, is_prefill_chunk=False))
    plan.entries.append(_admit(prefill_seq, 5))

    called = {"graphed": False}
    monkeypatch.setattr(
        graphed_llama_runner, "forward_graphed",
        lambda *a, **k: called.__setitem__("graphed", True),
    )
    graphed_llama_runner.forward(plan, seqs)
    assert not called["graphed"], "a plan with a prefill entry must never take the graphed path"

    allocator.free(decoding_seq.block_table)


def test_forward_dispatch_skips_graphs_above_the_largest_bucket(graphed_llama_runner, monkeypatch):
    from llm_serving_engine.batch_plan import BatchEntry, BatchPlan

    largest = max(graphed_llama_runner._decode_graphs)
    plan = BatchPlan()
    for i in range(largest + 1):
        plan.entries.append(BatchEntry(i, 1, is_prefill_chunk=False))

    called = {"graphed": False}
    monkeypatch.setattr(
        graphed_llama_runner, "forward_graphed",
        lambda *a, **k: called.__setitem__("graphed", True),
    )
    monkeypatch.setattr(graphed_llama_runner, "forward_fused", lambda *a, **k: [])
    graphed_llama_runner.forward(plan, {})
    assert not called["graphed"], "a plan larger than every captured bucket must fall back to forward_fused"


def test_sample_temperature_zero_is_argmax(loaded_runner):
    from llm_serving_engine.sampling import SamplingParams

    logits = torch.tensor([0.1, 5.0, -2.0, 0.3])
    token = loaded_runner._sample(logits, SamplingParams(temperature=0.0))
    assert token == 1


def test_sample_top_k_and_top_p_stay_in_range(loaded_runner):
    from llm_serving_engine.sampling import SamplingParams

    logits = torch.randn(50)
    for _ in range(20):
        token = loaded_runner._sample(logits, SamplingParams(top_k=5, top_p=0.9))
        assert 0 <= token < 50
