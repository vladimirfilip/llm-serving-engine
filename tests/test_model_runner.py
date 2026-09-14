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
        runner = ModelRunner(ModelConfig(model_name_or_path=MODEL, device="cpu"))
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


@pytest.fixture(scope="module")
def tiny_llama_runner():
    """attention_forward calls the CUDA-only FA-2 Triton kernel in kernels/flash_attention.py,
    so this exercises the real wiring end to end rather than standing in a fake. A random tiny
    LlamaConfig (GQA: 4 heads / 2 kv heads) needs no network or pretrained weights — it's the
    smallest thing shaped like the Llama-family decoder attention_forward assumes.
    """
    if not torch.cuda.is_available():
        pytest.skip("attention_forward calls a CUDA-only Triton kernel")
    pytest.importorskip("triton")

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
    runner.model = LlamaForCausalLM(hf_config).to("cuda").eval()
    return runner


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
    assert cache[seq_a.seq_id][0].shape[2] == 2
    assert cache[seq_b.seq_id][0].shape[2] == 4


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
