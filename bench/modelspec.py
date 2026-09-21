"""Model dimensions read from `config.json`, and the byte and FLOP counts derived from them."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DTYPE_BYTES = {"bfloat16": 2, "float16": 2, "float32": 4}


@dataclass(frozen=True, slots=True)
class ModelSpec:
    d: int  # hidden_size
    f: int  # intermediate_size
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    vocab: int
    dtype_bytes: int

    @classmethod
    def from_dir(cls, model_dir: str | Path, dtype: str) -> ModelSpec:
        cfg = json.loads((Path(model_dir).expanduser() / "config.json").read_text())
        return cls.from_config(cfg, dtype)

    @classmethod
    def from_config(cls, cfg: dict, dtype: str) -> ModelSpec:
        return cls(
            d=cfg["hidden_size"],
            f=cfg["intermediate_size"],
            n_layers=cfg["num_hidden_layers"],
            n_heads=cfg["num_attention_heads"],
            n_kv_heads=cfg["num_key_value_heads"],
            head_dim=cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"],
            vocab=cfg["vocab_size"],
            dtype_bytes=DTYPE_BYTES[dtype],
        )

    @property
    def layer_matmul_params(self) -> int:
        q_width, kv_width = self.n_heads * self.head_dim, self.n_kv_heads * self.head_dim
        return self.d * q_width + 2 * self.d * kv_width + q_width * self.d + 3 * self.d * self.f

    @property
    def norm_params(self) -> int:
        return (2 * self.n_layers + 1) * self.d

    @property
    def weight_bytes_decode(self) -> int:
        """Bytes one decode step reads. The LM head counts once, tied embeddings or not; the
        embedding lookup reads a single row and is left out."""
        params = self.n_layers * self.layer_matmul_params + self.vocab * self.d + self.norm_params
        return self.dtype_bytes * params

    @property
    def kv_bytes_per_tok_layer(self) -> int:
        return 2 * self.n_kv_heads * self.head_dim * self.dtype_bytes

    @property
    def kv_bytes_per_token(self) -> int:
        return self.n_layers * self.kv_bytes_per_tok_layer

    def bound_tok_s(self, ctx: float, bw_read_gbs: float) -> float:
        """Bandwidth-bound decode rate at batch 1: one pass over the weights and the context's KV."""
        return bw_read_gbs * 1e9 / (self.weight_bytes_decode + ctx * self.kv_bytes_per_token)

    def prefill_flops(self, prompt: int) -> float:
        """Causal prefill: matmuls, QK and PV over the triangle, LM head on the last position."""
        matmul = 2 * self.n_layers * self.layer_matmul_params * prompt
        attention = self.n_layers * 2 * prompt * prompt * self.n_heads * self.head_dim
        return matmul + attention + 2 * self.vocab * self.d
