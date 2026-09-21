"""Per-request sampling configuration and batched sampling."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0  # 0 is greedy
    top_p: float = 1.0
    top_k: int = -1  # -1 disables top-k
    max_tokens: int = 256
    ignore_eos: bool = False  # generate exactly max_tokens, past any stop token

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        if not 0 < self.top_p <= 1:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be > 0, got {self.max_tokens}")


def sample_tokens(logits: torch.Tensor, params: list[SamplingParams]) -> torch.Tensor:
    """(batch, vocab) logits -> (batch,) token ids on the same device, row i sampled under
    params[i]. The whole batch shares one softmax, at most one sort and one multinomial
    draw, and no host sync.

    Top-k keeps the k highest-probability tokens; top-p then keeps, within those, each
    token whose preceding probability mass is at most top_p of what top-k kept.
    """
    greedy = torch.argmax(logits, dim=-1)
    if all(p.temperature == 0 for p in params):
        return greedy

    vocab = logits.shape[-1]
    knobs = torch.tensor(
        [
            [p.temperature or 1.0, p.top_p, p.top_k if 0 < p.top_k < vocab else vocab]
            for p in params
        ],
        device=logits.device,
    )
    temperature, top_p, top_k = knobs[:, 0:1], knobs[:, 1:2], knobs[:, 2:3]
    probs = torch.softmax(logits.float() / temperature, dim=-1)

    if all(p.top_p == 1.0 and not 0 < p.top_k < vocab for p in params):
        sampled = torch.multinomial(probs, 1).squeeze(-1)
    else:
        sorted_probs, sorted_ids = torch.sort(probs, dim=-1, descending=True)
        rank = torch.arange(vocab, device=logits.device)
        sorted_probs = sorted_probs.masked_fill(rank >= top_k, 0.0)
        kept_mass = sorted_probs.sum(dim=-1, keepdim=True)
        mass_before = torch.cumsum(sorted_probs, dim=-1) - sorted_probs
        sorted_probs = sorted_probs.masked_fill(mass_before > top_p * kept_mass, 0.0)
        sampled = sorted_ids.gather(-1, torch.multinomial(sorted_probs, 1)).squeeze(-1)

    is_greedy = torch.tensor([p.temperature == 0 for p in params], device=logits.device)
    return torch.where(is_greedy, greedy, sampled)
