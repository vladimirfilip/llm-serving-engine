import pytest
import torch

from llm_serving_engine.model.sampling import SamplingParams, sample_tokens


def test_defaults_are_valid():
    p = SamplingParams()
    assert p.temperature == 1.0
    assert p.max_tokens == 256


@pytest.mark.parametrize(
    "kwargs", [{"temperature": -1}, {"top_p": 0}, {"top_p": 1.5}, {"max_tokens": 0}]
)
def test_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        SamplingParams(**kwargs)


def test_greedy_rows_take_the_argmax_beside_sampled_rows():
    logits = torch.tensor([[0.1, 5.0, -2.0], [3.0, 0.0, 0.0], [0.0, 0.0, 9.0]])
    greedy = SamplingParams(temperature=0.0)
    params = [greedy, SamplingParams(top_p=0.5), greedy]
    tokens = sample_tokens(logits, params).tolist()
    assert tokens[0] == 1
    assert tokens[2] == 2


def test_sampling_follows_the_softmax_distribution():
    logits = torch.log(torch.tensor([[0.25, 0.75]])).repeat(20000, 1)
    frequency = sample_tokens(logits, [SamplingParams()] * 20000).float().mean().item()
    assert frequency == pytest.approx(0.75, abs=0.02)


def test_temperature_sharpens_the_distribution():
    logits = torch.log(torch.tensor([[0.25, 0.75]])).repeat(20000, 1)
    frequency = sample_tokens(logits, [SamplingParams(temperature=0.5)] * 20000).float().mean()
    assert frequency.item() == pytest.approx(0.9, abs=0.02)  # 0.75^2 / (0.75^2 + 0.25^2)


def test_top_k_keeps_only_the_k_most_likely_tokens():
    logits = torch.log(torch.tensor([[0.4, 0.3, 0.2, 0.1]])).repeat(5000, 1)
    tokens = sample_tokens(logits, [SamplingParams(top_k=2)] * 5000)
    assert set(tokens.tolist()) == {0, 1}


def test_top_p_keeps_tokens_until_the_mass_before_them_passes_p():
    logits = torch.log(torch.tensor([[0.6, 0.3, 0.1]])).repeat(5000, 1)
    assert set(sample_tokens(logits, [SamplingParams(top_p=0.5)] * 5000).tolist()) == {0}
    assert set(sample_tokens(logits, [SamplingParams(top_p=0.8)] * 5000).tolist()) == {0, 1}


def test_top_p_applies_to_the_mass_top_k_kept():
    logits = torch.log(torch.tensor([[0.6, 0.3, 0.1]])).repeat(5000, 1)
    # top-k keeps 0.9 of the mass; token 1 follows 0.6 of it, past 0.6 * 0.9.
    params = [SamplingParams(top_k=2, top_p=0.6)] * 5000
    assert set(sample_tokens(logits, params).tolist()) == {0}


def test_rows_with_different_params_share_one_batch():
    logits = torch.log(torch.tensor([[0.4, 0.3, 0.2, 0.1]])).repeat(4000, 1)
    params = [SamplingParams(top_k=1), SamplingParams()] * 2000
    tokens = sample_tokens(logits, params)
    assert set(tokens[0::2].tolist()) == {0}
    assert set(tokens[1::2].tolist()) == {0, 1, 2, 3}
