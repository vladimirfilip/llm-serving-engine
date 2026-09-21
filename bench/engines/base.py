"""The contract every engine under test satisfies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


class CheckFailed(RuntimeError):
    """A MUST precondition failed: the run for that engine is aborted."""


@dataclass(frozen=True, slots=True)
class Launch:
    """What varies between launches of one engine: the tuned token budget and any ablation
    arguments or environment."""

    token_budget: int | None = None
    args_add: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    wrapper: list[str] = field(default_factory=list)  # command the server is launched under


class EngineAdapter(ABC):
    name: str
    accepts_token_ids: bool
    extra_body: dict
    itl_valid: bool = True  # cleared by the token-accounting and burst checks

    @abstractmethod
    def launch(self, launch: Launch, phase: str, env_overrides: dict | None = None) -> None: ...

    @abstractmethod
    def wait_ready(self, timeout_s: float = 900) -> float:
        """Seconds from spawn to the first successful completion."""

    @abstractmethod
    def base_url(self) -> str: ...

    @abstractmethod
    def stats(self) -> dict | None:
        """Running, waiting, preemptions and KV occupancy, or None when unavailable."""

    @abstractmethod
    def score(self, token_ids: list[int]) -> list[float] | None:
        """Logprob of each token given its prefix, `len(token_ids) - 1` values."""

    @abstractmethod
    def shutdown(self) -> None: ...


def request_body(model: str, prompt: list[int] | str, max_tokens: int, extra_body: dict,
                 stream: bool = True, logprobs: bool = False) -> dict:
    """The completion request every engine receives: greedy, exactly `max_tokens` tokens."""
    body = {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0,
            "stream": stream, **extra_body}
    if stream:
        body["stream_options"] = {"include_usage": True}
    if logprobs:
        body["logprobs"] = 1
    return body
