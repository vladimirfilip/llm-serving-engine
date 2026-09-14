"""Per-request sampling configuration and the sampling math itself."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1  # -1 disables top-k
    max_tokens: int = 256
    stop: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        if not 0 < self.top_p <= 1:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be > 0, got {self.max_tokens}")


def sample_token(logits: torch.Tensor, params: SamplingParams) -> torch.Tensor:
    """Returns a 0-dim device tensor, so a caller sampling a whole batch pays one host
    sync (`torch.stack(...).tolist()`) for all of it."""
    import torch

    if params.temperature == 0:
        return torch.argmax(logits)

    logits = logits / params.temperature
    if params.top_k > 0:
        top_k = min(params.top_k, logits.size(-1))
        kth_value = torch.topk(logits, top_k).values[..., -1]
        logits = torch.where(logits < kth_value, torch.full_like(logits, float("-inf")), logits)

    probs = torch.softmax(logits, dim=-1)
    if params.top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        drop = cumulative > params.top_p
        drop[..., 1:] = drop[..., :-1].clone()
        drop[..., 0] = False
        sorted_probs[drop] = 0.0
        probs = torch.zeros_like(probs).scatter(-1, sorted_idx, sorted_probs)
        probs = probs / probs.sum()

    return torch.multinomial(probs, 1).squeeze(0)

