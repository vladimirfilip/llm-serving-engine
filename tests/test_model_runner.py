"""ModelRunner's per-sequence HF path, on CPU."""

from __future__ import annotations

import pytest
import torch

from llm_serving_engine.config import ModelConfig
from llm_serving_engine.model import model_runner as model_runner_module
from llm_serving_engine.model.model_runner import ModelRunner, eos_token_ids
from llm_serving_engine.model.sampling import SamplingParams
from llm_serving_engine.scheduling.batch_plan import BatchEntry, BatchPlan
from tests.factories import admit, make_sequence


@pytest.fixture(scope="module")
def runner(tiny_gpt2_dir) -> ModelRunner:
    path = str(tiny_gpt2_dir)
    return ModelRunner(ModelConfig(model_name_or_path=path, device="cpu", use_custom_kernels=False))


def _greedy(seq_id: int, prompt: list[int], max_tokens: int = 8):
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    return make_sequence(seq_id=seq_id, prompt_tokens=prompt, sampling_params=params)


def _run(runner: ModelRunner, seq, *entries: BatchEntry):
    return runner.forward(BatchPlan(entries=list(entries)), {seq.seq_id: seq})


def _decode(runner: ModelRunner, seq):
    return _run(runner, seq, BatchEntry(seq.seq_id, 1, is_prefill_chunk=False))


def test_loads_on_cpu(runner):
    assert runner.device == "cpu"


def test_full_prefill_returns_a_token_and_caches_the_prompt(runner):
    seq = _greedy(1, [1, 2, 3, 4])

    [(seq_id, token, finished)] = _run(runner, seq, admit(seq, 4))

    assert seq_id == seq.seq_id
    assert 0 <= token < runner.model.config.vocab_size
    assert not finished
    assert seq.seq_id in runner._past_key_values
    runner.free(seq.seq_id)


def test_partial_prefill_chunk_samples_nothing(runner):
    seq = _greedy(2, [1, 2, 3, 4])
    assert _run(runner, seq, admit(seq, 2)) == []
    runner.free(seq.seq_id)


def test_decode_matches_a_non_incremental_forward(runner):
    seq = _greedy(3, [5, 6, 7])
    [(_, first, _)] = _run(runner, seq, admit(seq, 3))
    seq.generated_tokens.append(first)

    [(_, second, _)] = _decode(runner, seq)

    with torch.no_grad():
        logits = runner.model(torch.tensor([[5, 6, 7, first]])).logits
    assert second == int(torch.argmax(logits[0, -1]))
    runner.free(seq.seq_id)


def test_decode_finishes_at_max_tokens(runner):
    seq = _greedy(4, [1, 2], max_tokens=2)
    [(_, first, finished)] = _run(runner, seq, admit(seq, 2))
    assert not finished
    seq.generated_tokens.append(first)

    [(_, _, finished)] = _decode(runner, seq)
    assert finished
    runner.free(seq.seq_id)


def test_any_listed_eos_token_finishes_a_sequence(runner, monkeypatch):
    monkeypatch.setattr(runner, "eos_token_ids", eos_token_ids([999, 111]))
    monkeypatch.setattr(model_runner_module, "sample_token", lambda *_: torch.tensor(111))
    seq = _greedy(5, [1, 2], max_tokens=50)

    [(_, token, finished)] = _run(runner, seq, admit(seq, 2))

    assert token == 111
    assert finished
    runner.free(seq.seq_id)


def test_eos_token_ids_accepts_a_single_id_a_list_or_none():
    assert eos_token_ids(2) == {2}
    assert eos_token_ids([2, 3]) == {2, 3}
    assert eos_token_ids(None) == frozenset()


def test_a_preempted_sequence_recomputes_to_the_token_it_would_have_decoded(runner):
    seq = _greedy(6, [1, 2, 3])
    [(_, first, _)] = _run(runner, seq, admit(seq, 3))
    seq.generated_tokens.append(first)
    [(_, uninterrupted, _)] = _decode(runner, seq)
    runner.free(seq.seq_id)

    seq.prefill_progress, seq.status = 0, "WAITING"
    [(_, recomputed, _)] = _run(runner, seq, admit(seq, seq.num_tokens))

    assert recomputed == uninterrupted
    runner.free(seq.seq_id)


def test_free_drops_the_sequence_cache(runner):
    seq = _greedy(7, [1, 2])
    _run(runner, seq, admit(seq, 2))
    runner.free(seq.seq_id)
    assert seq.seq_id not in runner._past_key_values


def test_int8_quantization_is_not_implemented(tiny_gpt2_dir):
    with pytest.raises(NotImplementedError):
        ModelRunner(ModelConfig(model_name_or_path=str(tiny_gpt2_dir), quantize="int8"))


def test_custom_kernels_refuse_a_cpu_device(tiny_gpt2_dir):
    path = str(tiny_gpt2_dir)
    config = ModelConfig(model_name_or_path=path, device="cpu", use_custom_kernels=True)
    with pytest.raises(RuntimeError, match="CUDA"):
        ModelRunner(config)
