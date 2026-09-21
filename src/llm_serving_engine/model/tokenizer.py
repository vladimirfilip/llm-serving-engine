"""Tokenizer glue around HuggingFace `transformers.AutoTokenizer`."""

from __future__ import annotations

from transformers import AutoTokenizer

# What `decode` yields for bytes that end mid-character.
_INCOMPLETE_CHAR = "�"


class TokenizerWrapper:
    def __init__(self, model_name_or_path: str):
        self._tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        # seq_id -> (prefix_offset, read_offset) into that sequence's generated tokens.
        self._decode_windows: dict[int, tuple[int, int]] = {}

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text)

    def encode_prompt(self, prompt: str) -> list[int]:
        """Instruct checkpoints ship a chat template wrapping the prompt in the turn tokens
        they were tuned on; base checkpoints have none and continue the raw text."""
        if self._tokenizer.chat_template is None:
            return self.encode(prompt)
        return self._tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=False,
        )

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def decode_incremental(
        self, seq_id: int, generated_tokens: list[int], skip_special_tokens: bool = True
    ) -> str:
        """Text completed since the last call for this seq_id.

        Decodes only a window: tokens[prefix_offset:read_offset] were already emitted and
        give the merge context a token needs to decode with the right spacing. Text ending
        in an incomplete character is held back until the tokens that complete it arrive.
        """
        prefix_offset, read_offset = self._decode_windows.get(seq_id, (0, 0))
        prefix_text = self.decode(generated_tokens[prefix_offset:read_offset], skip_special_tokens)
        text = self.decode(generated_tokens[prefix_offset:], skip_special_tokens)
        if len(text) <= len(prefix_text) or text.endswith(_INCOMPLETE_CHAR):
            return ""
        self._decode_windows[seq_id] = (read_offset, len(generated_tokens))
        return text[len(prefix_text) :]

    def forget(self, seq_id: int) -> None:
        self._decode_windows.pop(seq_id, None)
