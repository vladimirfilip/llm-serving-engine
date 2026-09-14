import pytest

from llm_serving_engine.sampling import SamplingParams


def test_defaults_are_valid():
    p = SamplingParams()
    assert p.temperature == 1.0
    assert p.max_tokens == 256
    assert p.stop == []


@pytest.mark.parametrize("kwargs", [{"temperature": -1}, {"top_p": 0}, {"top_p": 1.5}, {"max_tokens": 0}])
def test_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        SamplingParams(**kwargs)


def test_stop_defaults_are_independent_lists():
    a, b = SamplingParams(), SamplingParams()
    a.stop.append("</s>")
    assert b.stop == []
