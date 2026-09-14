"""Per-request sampling configuration."""

from __future__ import annotations

from dataclasses import dataclass, field


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
