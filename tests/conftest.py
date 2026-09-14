from __future__ import annotations

from pathlib import Path

import pytest
import torch

from llm_serving_engine.scheduling.dispatch import output_channels


def pytest_runtest_setup(item: pytest.Item) -> None:
    """A CUDA test fails without CUDA, so a run can only pass by running it."""
    if item.get_closest_marker("cuda") and not torch.cuda.is_available():
        pytest.fail("needs CUDA; on a CPU-only machine run pytest -m 'not cuda'", pytrace=False)


@pytest.fixture(autouse=True)
def _clear_output_channels():
    output_channels.clear()
    yield
    output_channels.clear()


@pytest.fixture(scope="session")
def tiny_llama_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Random-weight Llama with GQA (4 query heads, 2 KV heads, head_dim 8), saved as a
    checkpoint so ModelRunner loads it like any model. No EOS token, so generation always
    runs to max_tokens."""
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
        bos_token_id=None, eos_token_id=None, pad_token_id=None,
    )
    path = tmp_path_factory.mktemp("tiny-llama")
    LlamaForCausalLM(config).save_pretrained(path)
    return path


@pytest.fixture(scope="session")
def tiny_gpt2_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Random-weight GPT-2: a model with no Llama-family layout."""
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    config = GPT2Config(vocab_size=64, n_positions=64, n_embd=16, n_layer=2, n_head=2)
    path = tmp_path_factory.mktemp("tiny-gpt2")
    GPT2LMHeadModel(config).save_pretrained(path)
    return path
