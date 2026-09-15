"""The `send_fn` the timing loops call: transport and per-request bookkeeping.

`send_request` reports what it observes locally (client-side first_token_latency, token
count, success/error). Overall request latency is measured only by the timing loop, so
exactly one clock defines it.
"""

from __future__ import annotations

import json
import random
import time

import httpx

from ..model.sampling import SamplingParams
from .timing import SendFn

DEFAULT_PROMPTS: list[str] = [
    "Explain the difference between a mutex and a semaphore.",
    "Write a haiku about garbage collection.",
    "What causes coordinated omission in load testing?",
    "Summarize continuous batching in two sentences.",
    "Describe how paged attention avoids memory fragmentation.",
]


def _parse_sse_line(line: str) -> dict | None:
    """Extract the JSON payload from one `data: {...}` SSE line, else None."""
    if not line.startswith("data:"):
        return None
    payload = line[len("data:") :].strip()
    if not payload or payload == "[DONE]":
        return None
    return json.loads(payload)


def load_client(
    base_url: str, transport: httpx.AsyncBaseTransport | None = None
) -> httpx.AsyncClient:
    """One client for a whole load run. Building a client loads a CA bundle, about 100 ms of
    CPU, so one per request would saturate the load generator before the server. Connections
    are unbounded, so no request ever waits in the client for a free one."""
    return httpx.AsyncClient(
        base_url=base_url,
        timeout=60.0,
        limits=httpx.Limits(max_connections=None, max_keepalive_connections=None),
        transport=transport,
    )


async def send_request(
    client: httpx.AsyncClient, prompt: str, sampling_params: SamplingParams | None = None
) -> dict:
    """POST /v1/generate, consume the SSE stream to completion, report request facts.
    Returns a dict the timing loop merges into its own result, so it never carries a
    "latency" key."""
    body: dict = {"prompt": prompt}
    if sampling_params is not None:
        body["max_tokens"] = sampling_params.max_tokens
        body["temperature"] = sampling_params.temperature
        body["top_p"] = sampling_params.top_p

    sent_at = time.monotonic()
    result: dict = {
        "success": True,
        "error": None,
        "num_tokens_received": 0,
        "first_token_latency": None,
        "token_times": [],
        "prompt_tokens": None,
        "output_tokens": None,
    }
    try:
        async with client.stream("POST", "/v1/generate", json=body) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                event = _parse_sse_line(line)
                if event is None:
                    continue
                if "error" in event:
                    return {**result, "success": False, "error": str(event["error"])}
                if "token" in event:
                    now = time.monotonic()
                    if result["first_token_latency"] is None:
                        result["first_token_latency"] = now - sent_at
                    result["token_times"].append(now)
                    result["num_tokens_received"] += 1
                if event.get("done"):
                    result["prompt_tokens"] = event.get("prompt_tokens")
                    result["output_tokens"] = event.get("output_tokens")
                    break
    except httpx.HTTPError as exc:
        # A timeout's message is empty, so the exception type is what identifies it.
        return {**result, "success": False, "error": f"{type(exc).__name__}: {exc}"}
    return result


def request_sender(
    client: httpx.AsyncClient,
    sampling_params: SamplingParams | None = None,
    prompts: list[str] = DEFAULT_PROMPTS,
) -> SendFn:
    """A send_fn whose every call sends one prompt drawn at random from `prompts`."""

    async def send() -> dict:
        return await send_request(client, random.choice(prompts), sampling_params)

    return send
