"""forward_fused on the Triton kernels, checked against HuggingFace's own forward on the
same weights: contiguous per-sequence buffers and the shared paged pool."""

from __future__ import annotations

import pytest
import torch

from llm_serving_engine.config import ModelConfig
from llm_serving_engine.model.model_runner import ModelRunner
from llm_serving_engine.model.sampling import SamplingParams
from llm_serving_engine.scheduling.allocator import BlockAllocator
from llm_serving_engine.scheduling.batch_plan import BatchEntry, BatchPlan
from tests.factories import admit, make_sequence

pytestmark = pytest.mark.cuda


def load_runner(path, **overrides) -> ModelRunner:
    fields = dict(model_name_or_path=str(path), device="cuda", dtype="float32")
    fields.update(overrides)
    return ModelRunner(ModelConfig(**fields))


@pytest.fixture(scope="module")
def contiguous_runner(tiny_llama_dir) -> ModelRunner:
    return load_runner(tiny_llama_dir)


@pytest.fixture(scope="module")
def paged_runner(tiny_llama_dir) -> ModelRunner:
    runner = load_runner(tiny_llama_dir)
    runner.allocate_kv_pool(num_blocks=64, block_size=4)
    return runner


def greedy(seq_id: int, prompt: list[int]):
    params = SamplingParams(temperature=0.0, max_tokens=8)
    return make_sequence(seq_id=seq_id, prompt_tokens=prompt, sampling_params=params)


def decode_entry(seq) -> BatchEntry:
    return BatchEntry(seq.seq_id, 1, is_prefill_chunk=False)


def reference_next_token(runner: ModelRunner, token_ids: list[int]) -> int:
    with torch.no_grad():
        logits = runner.model(torch.tensor([token_ids], device="cuda")).logits
    return int(torch.argmax(logits[0, -1]))


def layer0_inputs(runner: ModelRunner, plan: BatchPlan, seqs):
    input_ids, position_ids, offsets = runner._flatten_plan(plan, seqs)
    with torch.no_grad():
        hidden = runner.model.get_input_embeddings()(input_ids)
        position_embeddings = runner.model.model.rotary_emb(hidden, position_ids)
        normed = runner._layers[0].input_layernorm(hidden)
    return normed, position_embeddings, offsets


def hf_layer0_attention(runner: ModelRunner, token_ids: list[int]) -> torch.Tensor:
    """Layer-0 attention over the whole sequence in one causal, non-incremental call."""
    layer0 = runner._layers[0]
    n = len(token_ids)
    with torch.no_grad():
        hidden = runner.model.get_input_embeddings()(torch.tensor([token_ids], device="cuda"))
        position_ids = torch.arange(n, device="cuda").unsqueeze(0)
        position_embeddings = runner.model.model.rotary_emb(hidden, position_ids)
        causal_mask = torch.full((n, n), float("-inf"), device="cuda").triu(1)
        out, _ = layer0.self_attn(
            layer0.input_layernorm(hidden),
            position_embeddings=position_embeddings,
            attention_mask=causal_mask,
        )
    return out


def contiguous_layer0(runner: ModelRunner, plan: BatchPlan, seqs) -> torch.Tensor:
    normed, position_embeddings, offsets = layer0_inputs(runner, plan, seqs)
    with torch.no_grad():
        return runner.contiguous_attention(
            normed, position_embeddings, 0, plan=plan, seqs=seqs, offsets=offsets
        )


def test_llama_checkpoints_pass_the_kernel_support_check(contiguous_runner):
    contiguous_runner.check_custom_kernel_support()


def test_custom_kernels_refuse_a_non_llama_model(tiny_gpt2_dir):
    with pytest.raises(RuntimeError, match="Llama-family"):
        load_runner(tiny_gpt2_dir)


def test_custom_kernels_refuse_fp16_with_head_dim_under_16(tmp_path):
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(  # head_dim = 16 / 4 = 4
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=32,
    )
    LlamaForCausalLM(config).save_pretrained(tmp_path)
    with pytest.raises(RuntimeError, match="head_dim"):
        load_runner(tmp_path, dtype="float16")


def test_flatten_plan_concatenates_entries_with_their_true_positions(contiguous_runner):
    prefilling = greedy(20, [1, 2, 3, 4])
    prefilling.prefill_progress, prefilling.status = 3, "PREFILLING"
    decoding = greedy(21, [9])
    decoding.generated_tokens.append(42)
    plan = BatchPlan(entries=[BatchEntry(20, 3, is_prefill_chunk=True), decode_entry(decoding)])

    input_ids, position_ids, offsets = contiguous_runner._flatten_plan(
        plan, {20: prefilling, 21: decoding}
    )

    assert input_ids.tolist() == [[1, 2, 3, 42]]
    assert position_ids.tolist() == [[0, 1, 2, 1]]
    assert offsets == [(0, 3), (3, 4)]


def test_contiguous_attention_prefill_matches_hf(contiguous_runner):
    seq = greedy(30, [1, 2, 3, 4, 5])
    plan = BatchPlan(entries=[admit(seq, 5)])

    mine = contiguous_layer0(contiguous_runner, plan, {seq.seq_id: seq})

    assert torch.allclose(hf_layer0_attention(contiguous_runner, [1, 2, 3, 4, 5]), mine, atol=1e-3)


def test_contiguous_attention_decode_attends_to_the_whole_cache(contiguous_runner):
    seq = greedy(31, [3, 1, 4, 1, 5])
    seqs = {seq.seq_id: seq}
    contiguous_layer0(contiguous_runner, BatchPlan(entries=[admit(seq, 5)]), seqs)
    seq.generated_tokens.append(9)

    mine = contiguous_layer0(contiguous_runner, BatchPlan(entries=[decode_entry(seq)]), seqs)

    reference = hf_layer0_attention(contiguous_runner, [3, 1, 4, 1, 5, 9])
    assert torch.allclose(reference[:, -1:], mine, atol=1e-3)


def test_contiguous_attention_chunked_prefill_continues_the_cache(contiguous_runner):
    seq = greedy(32, [1, 2, 3, 4, 5, 6])
    seqs = {seq.seq_id: seq}
    contiguous_layer0(contiguous_runner, BatchPlan(entries=[admit(seq, 3)]), seqs)

    mine = contiguous_layer0(contiguous_runner, BatchPlan(entries=[admit(seq, 3)]), seqs)

    reference = hf_layer0_attention(contiguous_runner, [1, 2, 3, 4, 5, 6])
    assert torch.allclose(reference[:, 3:6], mine, atol=1e-3)


def test_contiguous_attention_keeps_one_buffer_per_sequence(contiguous_runner):
    seq_a, seq_b = greedy(33, [1, 1]), greedy(34, [2, 2, 2, 2])
    plan = BatchPlan(entries=[admit(seq_a, 2), admit(seq_b, 4)])

    contiguous_layer0(contiguous_runner, plan, {33: seq_a, 34: seq_b})

    cache = contiguous_runner._kv_cache[0]
    assert cache[33][2] == 2  # filled length
    assert cache[34][2] == 4


def test_contiguous_forward_fused_prefill_then_decode_matches_hf(contiguous_runner):
    prompt = [3, 1, 4, 1, 5]
    seq = greedy(40, prompt)
    seqs = {seq.seq_id: seq}

    plan = BatchPlan(entries=[admit(seq, 5)])
    [(_, first, finished)] = contiguous_runner.forward_fused(plan, seqs)
    assert not finished
    assert first == reference_next_token(contiguous_runner, prompt)

    seq.generated_tokens.append(first)
    plan = BatchPlan(entries=[decode_entry(seq)])
    [(_, second, _)] = contiguous_runner.forward_fused(plan, seqs)
    assert second == reference_next_token(contiguous_runner, [*prompt, first])


def test_paged_forward_fused_prefill_then_decode_matches_hf(paged_runner):
    allocator = BlockAllocator(num_blocks=64, block_size=4)
    prompt = [3, 1, 4, 1, 5]
    seq = greedy(50, prompt)
    seqs = {seq.seq_id: seq}
    assert allocator.allocate(seq, len(prompt))

    [(_, first, _)] = paged_runner.forward_fused(BatchPlan(entries=[admit(seq, 5)]), seqs)
    assert first == reference_next_token(paged_runner, prompt)

    seq.generated_tokens.append(first)
    assert allocator.allocate(seq, 1)
    [(_, second, _)] = paged_runner.forward_fused(BatchPlan(entries=[decode_entry(seq)]), seqs)
    assert second == reference_next_token(paged_runner, [*prompt, first])


def test_paged_sequences_sharing_the_pool_never_see_each_other(paged_runner):
    allocator = BlockAllocator(num_blocks=64, block_size=4)
    prompt_a, prompt_b = [1, 1, 3], [2, 2, 2, 2, 2]
    seq_a, seq_b = greedy(51, prompt_a), greedy(52, prompt_b)
    assert allocator.allocate(seq_a, len(prompt_a))
    assert allocator.allocate(seq_b, len(prompt_b))
    plan = BatchPlan(entries=[admit(seq_a, 3), admit(seq_b, 5)])

    result_a, result_b = paged_runner.forward_fused(plan, {51: seq_a, 52: seq_b})

    assert result_a[1] == reference_next_token(paged_runner, prompt_a)
    assert result_b[1] == reference_next_token(paged_runner, prompt_b)


def test_paged_decode_and_prefill_share_one_iteration(paged_runner):
    allocator = BlockAllocator(num_blocks=64, block_size=4)
    prompt_a, prompt_b = [4, 4, 4], [6, 6, 6, 6]
    seq_a, seq_b = greedy(53, prompt_a), greedy(54, prompt_b)
    seqs = {53: seq_a, 54: seq_b}
    assert allocator.allocate(seq_a, 3)
    [(_, first, _)] = paged_runner.forward_fused(BatchPlan(entries=[admit(seq_a, 3)]), seqs)
    seq_a.generated_tokens.append(first)
    assert allocator.allocate(seq_a, 1)
    assert allocator.allocate(seq_b, 4)

    result_a, result_b = paged_runner.forward_fused(
        BatchPlan(entries=[decode_entry(seq_a), admit(seq_b, 4)]), seqs
    )

    assert result_a[1] == reference_next_token(paged_runner, [*prompt_a, first])
    assert result_b[1] == reference_next_token(paged_runner, prompt_b)


def test_paged_recompute_after_preemption_yields_the_uninterrupted_token(paged_runner):
    allocator = BlockAllocator(num_blocks=64, block_size=4)
    seq = greedy(55, [7, 3, 5])
    seqs = {seq.seq_id: seq}
    assert allocator.allocate(seq, 3)
    [(_, first, _)] = paged_runner.forward_fused(BatchPlan(entries=[admit(seq, 3)]), seqs)
    seq.generated_tokens.append(first)
    assert allocator.allocate(seq, 1)
    plan = BatchPlan(entries=[decode_entry(seq)])
    [(_, uninterrupted, _)] = paged_runner.forward_fused(plan, seqs)

    allocator.free(seq.block_table)
    paged_runner.free(seq.seq_id)
    seq.prefill_progress, seq.status = 0, "WAITING"
    assert allocator.allocate(seq, seq.num_tokens)
    [(_, recomputed, _)] = paged_runner.forward_fused(
        BatchPlan(entries=[admit(seq, seq.num_tokens)]), seqs
    )

    assert recomputed == uninterrupted


def test_kv_pool_has_one_scratch_block_past_the_allocator_ids(paged_runner):
    config = paged_runner.model.config
    expected = (65, 4, config.num_key_value_heads, config.hidden_size // config.num_attention_heads)
    assert len(paged_runner._k_pool) == config.num_hidden_layers
    assert tuple(paged_runner._k_pool[0].shape) == expected
    assert tuple(paged_runner._v_pool[0].shape) == expected
    assert paged_runner._scratch_block_id == 64
