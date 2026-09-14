from __future__ import annotations

import pytest

transformers = pytest.importorskip("transformers")
pytest.importorskip("torch")

MODEL = "sshleifer/tiny-gpt2"


@pytest.fixture(scope="module")
def wrapper():
    from llm_serving_engine.tokenizer import TokenizerWrapper

    try:
        return TokenizerWrapper(MODEL)
    except Exception as e:
        pytest.skip(f"can't fetch {MODEL} from the HF Hub: {e}")


def test_encode_decode_roundtrip(wrapper):
    ids = wrapper.encode("Hello, world!")
    assert ids and all(isinstance(i, int) for i in ids)
    assert isinstance(wrapper.decode(ids), str)


def test_eos_token_id_is_int(wrapper):
    assert isinstance(wrapper.eos_token_id, int)


def test_decode_incremental_matches_full_decode(wrapper):
    ids = wrapper.encode("The quick brown fox jumps over the lazy dog")
    pieces = []
    for i in range(1, len(ids) + 1):
        pieces.append(wrapper.decode_incremental(seq_id=1, generated_tokens=ids[:i]))
    assert "".join(pieces) == wrapper.decode(ids)
    wrapper.forget(1)


def test_decode_incremental_is_per_sequence(wrapper):
    ids_a = wrapper.encode("apples")
    ids_b = wrapper.encode("oranges")
    first_a = wrapper.decode_incremental(seq_id=10, generated_tokens=ids_a[:1])
    wrapper.decode_incremental(seq_id=20, generated_tokens=ids_b[:1])
    # Interleaved sequences don't see each other's prior decoded text.
    assert wrapper.decode_incremental(seq_id=10, generated_tokens=ids_a) == wrapper.decode(
        ids_a
    )[len(first_a) :]
    wrapper.forget(10)
    wrapper.forget(20)


def test_encode_prompt_falls_back_to_raw_encode_without_chat_template(wrapper):
    # tiny-gpt2 is a base-style tokenizer with no chat template.
    assert wrapper._tokenizer.chat_template is None
    assert wrapper.encode_prompt("Hello, world!") == wrapper.encode("Hello, world!")


def test_encode_prompt_applies_chat_template_when_present(wrapper, monkeypatch):
    monkeypatch.setattr(wrapper._tokenizer, "chat_template", "{# fake instruct template #}")
    calls = []

    def fake_apply(messages, add_generation_prompt, tokenize, return_dict):
        calls.append((messages, add_generation_prompt, tokenize, return_dict))
        return [1, 2, 3]

    monkeypatch.setattr(wrapper._tokenizer, "apply_chat_template", fake_apply, raising=False)
    assert wrapper.encode_prompt("hi") == [1, 2, 3]
    assert calls == [([{"role": "user", "content": "hi"}], True, True, False)]


def test_forget_resets_stream_state(wrapper):
    ids = wrapper.encode("hello")
    wrapper.decode_incremental(seq_id=99, generated_tokens=ids)
    wrapper.forget(99)
    # After forgetting, the next call sees no prior text, so it returns the full decode again.
    assert wrapper.decode_incremental(seq_id=99, generated_tokens=ids) == wrapper.decode(ids)
    wrapper.forget(99)
