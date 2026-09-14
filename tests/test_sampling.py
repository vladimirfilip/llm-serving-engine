import pytest
import torch

from llm_serving_engine.model.sampling import SamplingParams, sample_token


def test_defaults_are_valid():
    p = SamplingParams()
    assert p.temperature == 1.0
    assert p.max_tokens == 256
    assert p.stop == []


@pytest.mark.parametrize(
    "kwargs", [{"temperature": -1}, {"top_p": 0}, {"top_p": 1.5}, {"max_tokens": 0}]
)
def test_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        SamplingParams(**kwargs)


def test_stop_defaults_are_independent_lists():
    a, b = SamplingParams(), SamplingParams()
    a.stop.append("</s>")
    assert b.stop == []


def test_temperature_zero_is_argmax():
    logits = torch.tensor([0.1, 5.0, -2.0, 0.3])
    assert int(sample_token(logits, SamplingParams(temperature=0.0))) == 1


def test_top_k_and_top_p_sample_only_from_the_head_of_the_distribution():
    logits = torch.linspace(5.0, 0.0, 50)  # token 0 most likely
    params = SamplingParams(top_k=5, top_p=0.99)
    assert {int(sample_token(logits, params)) for _ in range(50)} <= {0, 1, 2, 3, 4}
