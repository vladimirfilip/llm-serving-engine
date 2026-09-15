from __future__ import annotations

import pytest

from llm_serving_engine.model.tokenizer import TokenizerWrapper

MODEL = "sshleifer/tiny-gpt2"


@pytest.fixture(scope="module")
def wrapper() -> TokenizerWrapper:
    return TokenizerWrapper(MODEL)


def stream(wrapper: TokenizerWrapper, seq_id: int, ids: list[int]) -> list[str]:
    pieces = [wrapper.decode_incremental(seq_id, ids[:i]) for i in range(1, len(ids) + 1)]
    wrapper.forget(seq_id)
    return pieces


def test_encode_decode_roundtrip(wrapper):
    ids = wrapper.encode("Hello, world!")
    assert wrapper.decode(ids) == "Hello, world!"


def test_streamed_pieces_join_to_the_full_decode(wrapper):
    ids = wrapper.encode("The quick brown fox jumps over the lazy dog")
    assert "".join(stream(wrapper, 1, ids)) == wrapper.decode(ids)


def test_a_character_split_across_tokens_streams_whole(wrapper):
    ids = wrapper.encode("llama 🦙 ok 日本語")
    pieces = stream(wrapper, 2, ids)
    assert not any("�" in piece for piece in pieces)
    assert "".join(pieces) == "llama 🦙 ok 日本語"


def test_interleaved_sequences_keep_separate_decode_state(wrapper):
    ids_a = wrapper.encode("apples and pears")
    ids_b = wrapper.encode("oranges")
    pieces_a = [wrapper.decode_incremental(10, ids_a[:1])]
    wrapper.decode_incremental(20, ids_b[:1])
    pieces_a += [wrapper.decode_incremental(10, ids_a[:i]) for i in range(2, len(ids_a) + 1)]
    assert "".join(pieces_a) == wrapper.decode(ids_a)
    wrapper.forget(10)
    wrapper.forget(20)


def test_forget_restarts_the_stream(wrapper):
    ids = wrapper.encode("hello")
    wrapper.decode_incremental(99, ids)
    wrapper.forget(99)
    assert wrapper.decode_incremental(99, ids) == wrapper.decode(ids)
    wrapper.forget(99)


def test_encode_prompt_encodes_raw_text_without_a_chat_template(wrapper):
    assert wrapper._tokenizer.chat_template is None
    assert wrapper.encode_prompt("Hello, world!") == wrapper.encode("Hello, world!")


def test_encode_prompt_applies_a_chat_template_when_present(wrapper, monkeypatch):
    monkeypatch.setattr(wrapper._tokenizer, "chat_template", "{# instruct template #}")
    calls = []

    def fake_apply(messages, add_generation_prompt, tokenize, return_dict):
        calls.append((messages, add_generation_prompt, tokenize, return_dict))
        return [1, 2, 3]

    monkeypatch.setattr(wrapper._tokenizer, "apply_chat_template", fake_apply, raising=False)
    assert wrapper.encode_prompt("hi") == [1, 2, 3]
    assert calls == [([{"role": "user", "content": "hi"}], True, True, False)]
