"""The `send_fn` the timing loops call: transport and per-request bookkeeping.

`send_request` reports what it observes: when each token arrived, token counts,
success/error. Every latency, time to first token included, is computed by the timing loop
from its own send time, so one clock defines them all.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, replace

import httpx

from ..model.sampling import SamplingParams
from .timing import SendFn

CHAT_PROMPTS: list[str] = [
    "Explain the difference between a mutex and a semaphore.",
    "Write a haiku about garbage collection.",
    "What causes coordinated omission in load testing?",
    "Summarize continuous batching in two sentences.",
    "Describe how paged attention avoids memory fragmentation.",
]


@dataclass(slots=True)
class RequestShape:
    """One kind of request in a workload; `weight` is its share of requests sent."""

    name: str
    prompt: str
    max_tokens: int
    weight: float


def worker_log(lines: int) -> str:
    """A synthetic log of `lines` distinct lines, 19 Llama-3 tokens each."""
    return "".join(
        f"{i:04d} worker-{i % 7} finished batch {(i * 37) % 1000} in {(i * 13) % 97} ms, "
        f"queue depth {(i * 11) % 64}\n"
        for i in range(lines)
    )


# Mostly short chat turns. The document shapes' long prompts and outputs hold far more KV per
# request, so under load the running sequences' caches press against the pool's capacity.
WORKLOAD: list[RequestShape] = [
    *(RequestShape("chat", prompt, max_tokens=64, weight=0.15) for prompt in CHAT_PROMPTS),
    RequestShape(  # ~1000 prompt tokens
        "document",
        "Summarize the anomalies in this worker log.\n" + worker_log(52),
        max_tokens=256,
        weight=0.18,
    ),
    RequestShape(  # ~3000 prompt tokens
        "long_document",
        "List every batch slower than 90 ms in this worker log, with its worker.\n"
        + worker_log(157),
        max_tokens=512,
        weight=0.05,
    ),
    RequestShape(  # ~9000 prompt tokens: over two default 4096-token budgets, so 3+ chunks
        "chunked_document",
        "Which worker has the highest total batch time in this log?\n" + worker_log(473),
        max_tokens=256,
        weight=0.02,
    ),
]

# Requests of every shape a run sends, however small its weight: a shape seen a handful of
# times has no p90 worth reading.
MIN_REQUESTS_PER_SHAPE = 20


def _parse_sse_line(line: str) -> dict | None:
    """Extract the JSON payload from one `data: {...}` SSE line, else None."""
    if not line.startswith("data:"):
        return None
    payload = line[len("data:") :].strip()
    if not payload or payload == "[DONE]":
        return None
    return json.loads(payload)


OPEN_LOOP_TIMEOUT_S = 60.0


def load_client(
    base_url: str,
    timeout_s: float | None = OPEN_LOOP_TIMEOUT_S,
    transport: httpx.AsyncBaseTransport | None = None,
) -> httpx.AsyncClient:
    """One client for a whole load run. Building a client loads a CA bundle, about 100 ms of
    CPU, so one per request would saturate the load generator before the server. Connections
    are unbounded, so no request ever waits in the client for a free one.

    An open loop needs `timeout_s`: past capacity its backlog grows without bound. A closed
    loop passes None: its clients bound the backlog, so a request queued behind them is slow,
    not failed."""
    return httpx.AsyncClient(
        base_url=base_url,
        timeout=timeout_s,
        limits=httpx.Limits(max_connections=None, max_keepalive_connections=None),
        transport=transport,
    )


async def send_request(
    client: httpx.AsyncClient, prompt: str, sampling_params: SamplingParams | None = None
) -> dict:
    """POST /v1/generate, consume the SSE stream to completion, report request facts.
    `token_times` holds each token's arrival on `time.monotonic()`; the timing loop turns
    them into latencies, so the returned dict carries none."""
    body: dict = {"prompt": prompt}
    if sampling_params is not None:
        body["max_tokens"] = sampling_params.max_tokens
        body["temperature"] = sampling_params.temperature
        body["top_p"] = sampling_params.top_p

    result: dict = {
        "success": True,
        "error": None,
        "num_tokens_received": 0,
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
                    result["token_times"].append(time.monotonic())
                    result["num_tokens_received"] += 1
                if event.get("done"):
                    result["prompt_tokens"] = event.get("prompt_tokens")
                    result["output_tokens"] = event.get("output_tokens")
                    break
    except httpx.HTTPError as exc:
        # A timeout's message is empty, so the exception type is what identifies it.
        return {**result, "success": False, "error": f"{type(exc).__name__}: {exc}"}
    return result


def shape_schedule(
    workload: list[RequestShape],
    num_requests: int,
    rng: random.Random,
    min_per_shape: int = MIN_REQUESTS_PER_SHAPE,
) -> list[RequestShape]:
    """`num_requests` shapes in send order: each shape takes its share of the workload's
    weight, but never fewer than `min_per_shape`, so the rarest shape's percentiles rest on
    a sample instead of on however many a weighted draw happened to produce. Shuffled with
    `rng`, so the shapes interleave as arrivals rather than running shape by shape.

    The floors together never take more than half a run, so a run too small to hold them
    keeps the workload's proportions rather than becoming an even split of its shapes."""
    floor = min(min_per_shape, num_requests // (2 * len(workload)))
    weight_total = sum(shape.weight for shape in workload)
    by_weight = num_requests - floor * len(workload)
    exact = [floor + shape.weight / weight_total * by_weight for shape in workload]
    counts = [int(share) for share in exact]
    # Largest remainder, so the counts sum to num_requests without a shape losing its floor.
    ranked = sorted(range(len(exact)), key=lambda i: exact[i] - counts[i], reverse=True)
    for i in ranked[: num_requests - sum(counts)]:
        counts[i] += 1
    schedule = [shape for shape, count in zip(workload, counts, strict=True) for _ in range(count)]
    rng.shuffle(schedule)
    return schedule


def request_sender(
    client: httpx.AsyncClient,
    shapes: list[RequestShape],
    sampling_params: SamplingParams | None = None,
) -> SendFn:
    """A send_fn whose calls send `shapes` in order, one per call, each generating up to its
    shape's max_tokens with `sampling_params`' other settings. Each result names its shape,
    so raw results split by request size."""
    base = sampling_params or SamplingParams()
    queued = iter(shapes)

    async def send() -> dict:
        shape = next(queued)
        params = replace(base, max_tokens=shape.max_tokens)
        return {"shape": shape.name, **await send_request(client, shape.prompt, params)}

    return send
