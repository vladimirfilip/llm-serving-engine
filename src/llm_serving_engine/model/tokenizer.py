"""Tokenizer glue around HuggingFace `transformers.AutoTokenizer`.

Lazy-imports transformers so the rest of the package (allocator, scheduler, dispatch)
stays importable on a machine without it installed.
"""

from __future__ import annotations


class TokenizerWrapper:
    def __init__(self, model_name_or_path: str):
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self._stream_text: dict[int, str] = {}

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text)

    def encode_prompt(self, prompt: str) -> list[int]:
        """Encodes a raw `/v1/generate` prompt string for whichever checkpoint is loaded.

        Instruct checkpoints (Llama-3-Instruct and similar) ship a chat template that
        wraps the prompt in the header/turn tokens they were tuned on; base checkpoints
        have none and expect the raw text continued as-is. Presence of a chat template is
        what tells the two apart, so this branches on that instead of a config knob.
        """
        if self._tokenizer.chat_template is None:
            return self.encode(prompt)
        return self._tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=False,
        )

    def decode(self, token_ids: list[int]) -> str:
        return self._tokenizer.decode(token_ids, skip_special_tokens=True)

    def decode_incremental(self, seq_id: int, generated_tokens: list[int]) -> str:
        """Text newly produced since the last call for this seq_id.

        BPE/sentencepiece merges mean a single new token decoded in isolation can come out
        with the wrong leading-space/joining behavior, so this re-decodes the whole
        (prompt-free) generated-token list each call and diffs against the previous decode,
        rather than decoding the new token id alone. Still cheap: generated_tokens, not the
        prompt, is what grows per step.
        """
        full_text = self.decode(generated_tokens)
        prev_text = self._stream_text.get(seq_id, "")
        self._stream_text[seq_id] = full_text
        return full_text[len(prev_text) :]

    def forget(self, seq_id: int) -> None:
        """Drops incremental-decode state for a finished/cancelled sequence."""
        self._stream_text.pop(seq_id, None)

    @property
    def eos_token_id(self) -> int | None:
        return self._tokenizer.eos_token_id
