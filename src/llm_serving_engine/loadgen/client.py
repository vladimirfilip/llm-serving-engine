"""The `send_fn` that open_loop_load_gen calls: transport and per-request bookkeeping.

`send_request` reports what it observes locally (client-side first_token_latency, token
count, success/error). Overall request latency is measured only by the timing loop, from
`intended_send_time`, so exactly one clock defines it.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field

import httpx

from ..model.sampling import SamplingParams

DEFAULT_PROMPTS: list[str] = [
    "Explain the difference between a mutex and a semaphore.",
    "Write a haiku about garbage collection.",
    "What causes coordinated omission in load testing?",
    "Summarize continuous batching in two sentences.",
    "Describe how paged attention avoids memory fragmentation.",
]


@dataclass(slots=True)
class LoadGenConfig:
    target_qps: float
    duration_s: float
    base_url: str = "http://127.0.0.1:8000"
    prompts: list[str] = field(default_factory=lambda: list(DEFAULT_PROMPTS))
    sampling_params: SamplingParams | None = None

    def sample_prompt(self) -> str:
        return random.choice(self.prompts)


def _parse_sse_line(line: str) -> dict | None:
    """Extract the JSON payload from one `data: {...}` SSE line, else None."""
    if not line.startswith("data:"):
        return None
    payload = line[len("data:") :].strip()
    if not payload or payload == "[DONE]":
        return None
    return json.loads(payload)


async def send_request(
    base_url: str,
    prompt: str,
    sampling_params: SamplingParams | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> dict:
    """POST /v1/generate, consume the SSE stream to completion, report request facts.

    Returns a dict the timing loop merges into its own result, so it never carries a
    "latency" key. `transport` lets
    tests substitute an httpx.MockTransport for a real connection.
    """
    body: dict = {"prompt": prompt}
    if sampling_params is not None:
        body["max_tokens"] = sampling_params.max_tokens
        body["temperature"] = sampling_params.temperature
        body["top_p"] = sampling_params.top_p

    sent_at = time.monotonic()
    first_token_latency: float | None = None
    num_tokens_received = 0
    token_times: list[float] = []
    prompt_tokens: int | None = None
    output_tokens: int | None = None

    try:
        async with httpx.AsyncClient(base_url=base_url, timeout=60.0, transport=transport) as client:
            async with client.stream("POST", "/v1/generate", json=body) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    event = _parse_sse_line(line)
                    if event is None:
                        continue
                    if "error" in event:
                        return {
                            "success": False,
                            "error": str(event["error"]),
                            "num_tokens_received": num_tokens_received,
                            "first_token_latency": first_token_latency,
                            "token_times": token_times,
                            "prompt_tokens": prompt_tokens,
                            "output_tokens": output_tokens,
                        }
                    if "token" in event:
                        now = time.monotonic()
                        if first_token_latency is None:
                            first_token_latency = now - sent_at
                        token_times.append(now)
                        num_tokens_received += 1
                    if event.get("done"):
                        prompt_tokens = event.get("prompt_tokens")
                        output_tokens = event.get("output_tokens")
                        break
        return {
            "success": True,
            "error": None,
            "num_tokens_received": num_tokens_received,
            "first_token_latency": first_token_latency,
            "token_times": token_times,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
        }
    except httpx.HTTPError as exc:
        return {
            "success": False,
            "error": str(exc),
            "num_tokens_received": num_tokens_received,
            "first_token_latency": first_token_latency,
            "token_times": token_times,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
        }
