"""The engine's hooks for the benchmark harness: token-id submission, per-token logprobs, the
stats snapshot, the device-memory cap and NVTX ranges."""

import importlib
import types

import pytest
import torch

from llm_serving_engine import engine as engine_module
from llm_serving_engine.config import EngineConfig, KVCacheConfig
from llm_serving_engine.engine import InvalidPrompt
from llm_serving_engine.model.model_runner import ModelRunner
from llm_serving_engine.model.sampling import SamplingParams
from llm_serving_engine.observability import nvtx
from llm_serving_engine.scheduling.dispatch import output_channels
from tests.factories import decoding_sequence, make_sequence
from tests.test_engine import make_engine


def test_submit_tokens_sends_the_ids_untouched_and_skips_the_tokenizer():
    engine = make_engine()
    submission = engine.submit_tokens([5, 6, 7], SamplingParams())

    req = engine.ingress.get_nowait()
    assert req.prompt_tokens == [5, 6, 7] and submission.prompt_len == 3
    assert req.logprobs is None and submission.logprobs is None
    assert output_channels[submission.seq_id] is submission.output_queue


def test_a_logprob_request_shares_one_list_between_the_stream_and_the_sequence():
    engine = make_engine()
    submission = engine.submit_tokens([5, 6], SamplingParams(), logprobs=True)
    seq = engine_module.sequence_from_ingress(engine.ingress.get_nowait())
    assert seq.logprobs is submission.logprobs == []


@pytest.mark.parametrize("prompt", [[], [999]])
def test_submit_tokens_rejects_prompts_the_model_cannot_run(prompt):
    with pytest.raises(InvalidPrompt):
        make_engine().submit_tokens(prompt, SamplingParams())


def test_logprobs_are_the_chosen_tokens_log_softmax_and_only_for_requesting_sequences():
    logits = torch.tensor([[2.0, 1.0, 0.0, -1.0], [0.0, 0.0, 5.0, 0.0], [1.0, 2.0, 3.0, 4.0]])
    sampled = torch.tensor([0, 2, 3])
    wants = make_sequence(seq_id=1, logprobs=[])
    declines = make_sequence(seq_id=2)
    wants_too = make_sequence(seq_id=3, logprobs=[0.5])  # earlier tokens are kept
    ModelRunner._record_logprobs(logits, sampled, [wants, declines, wants_too])

    expected = logits.log_softmax(-1)
    assert wants.logprobs == pytest.approx([expected[0, 0].item()], abs=1e-5)
    assert declines.logprobs is None
    assert wants_too.logprobs == pytest.approx([0.5, expected[2, 3].item()], abs=1e-5)


def test_emit_tokens_records_the_logprob_before_returning_the_token():
    runner = object.__new__(ModelRunner)
    runner.eos_token_ids = frozenset()
    logits = torch.tensor([[0.0, 3.0, 0.0], [9.0, 0.0, 0.0]])  # row 1 is a graph's padding row
    seq = make_sequence(sampling_params=SamplingParams(temperature=0.0), logprobs=[])
    results = runner.emit_tokens(logits, [seq])
    assert results == [(seq.seq_id, 1, False)]
    assert seq.logprobs == pytest.approx([logits[0].log_softmax(-1)[1].item()], abs=1e-5)


def test_stats_report_occupancy_kv_blocks_and_the_runners_memory_split():
    engine = make_engine()
    seq = decoding_sequence(engine.allocator, seq_id=1, prompt_len=20)
    engine.running.append(seq)
    engine.waiting.append(make_sequence(seq_id=2))
    engine._preemptions = 3

    stats = engine.stats()
    assert (stats["running"], stats["waiting"], stats["preemptions_total"]) == (1, 1, 3)
    blocks_used = -(-20 // engine.config.kv_cache.block_size)
    assert stats["kv"] == {"block_size": engine.config.kv_cache.block_size,
                           "blocks_total": engine.allocator.num_blocks,
                           "blocks_used": blocks_used, "tokens_used": 20}
    assert sum(stats["memory_bytes"].values()) == 38


def test_stats_size_a_contiguous_pool_in_blocks():
    config = EngineConfig(**{**make_engine().config.__dict__, "kv_allocator": "contiguous"})
    engine = make_engine(config)
    assert engine.stats()["kv"]["blocks_total"] == (
        engine.allocator.capacity_tokens // config.kv_cache.block_size
    )


@pytest.mark.parametrize(
    ("fraction", "expected"),
    [(None, 6.0), (0.9, 5.0), (0.75, 3.5), (0.6, 2.0)],
)
def test_the_device_memory_cap_limits_the_pool_to_what_the_fraction_leaves(
    monkeypatch, fraction, expected
):
    gib = 1024**3
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (6 * gib, 10 * gib))
    runner = types.SimpleNamespace(device="cuda")
    assert engine_module._capped_free_bytes(runner, fraction) == expected * gib


def test_device_memory_fraction_comes_from_the_environment(monkeypatch):
    assert KVCacheConfig.from_env().device_memory_fraction is None
    monkeypatch.setenv("LLM_DEVICE_MEM_UTIL", "0.9")
    assert KVCacheConfig.from_env().device_memory_fraction == 0.9


def test_nvtx_ranges_are_one_shared_no_op_unless_enabled():
    assert not nvtx.ENABLED
    assert nvtx.nvtx_range("step") is nvtx.nvtx_range("forward")
    with nvtx.nvtx_range("step"):
        pass


def test_enabled_nvtx_pushes_and_pops_each_named_range(monkeypatch):
    calls = []
    monkeypatch.setenv("ENGINE_NVTX", "1")
    monkeypatch.setattr(torch.cuda.nvtx, "range_push", lambda name: calls.append(("push", name)))
    monkeypatch.setattr(torch.cuda.nvtx, "range_pop", lambda: calls.append(("pop",)))
    enabled = importlib.reload(nvtx)
    try:
        with enabled.nvtx_range("step"), enabled.nvtx_range("schedule"):
            pass
        assert calls == [("push", "step"), ("push", "schedule"), ("pop",), ("pop",)]
    finally:
        monkeypatch.delenv("ENGINE_NVTX")
        importlib.reload(nvtx)
