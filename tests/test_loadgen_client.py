import random
from collections import Counter

import httpx
import pytest

from llm_serving_engine.loadgen.client import (
    WORKLOAD,
    RequestShape,
    load_client,
    request_sender,
    send_request,
    shape_schedule,
)
from llm_serving_engine.model.sampling import SamplingParams


def _sse_client(body: str, status_code: int = 200) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        headers = {"content-type": "text/event-stream"}
        return httpx.Response(status_code, content=body.encode(), headers=headers)

    return load_client("http://test", transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_send_request_timestamps_every_token_and_leaves_latencies_to_the_timing_loop():
    body = (
        'data: {"token": "Hello"}\n\ndata: {"token": " world"}\n\n'
        'data: {"done": true, "prompt_tokens": 5, "output_tokens": 2}\n\n'
    )
    async with _sse_client(body) as client:
        result = await send_request(client, "hi")

    assert result["success"] is True
    assert result["error"] is None
    assert result["num_tokens_received"] == 2
    assert "latency" not in result and "first_token_latency" not in result
    assert result["prompt_tokens"] == 5
    assert result["output_tokens"] == 2
    assert result["token_times"] == sorted(result["token_times"])
    assert len(result["token_times"]) == 2


@pytest.mark.asyncio
async def test_send_request_reports_a_stream_level_error():
    async with _sse_client('data: {"error": "generation failed"}\n\n') as client:
        result = await send_request(client, "hi")
    assert result["success"] is False
    assert "generation failed" in result["error"]


@pytest.mark.asyncio
async def test_send_request_reports_an_http_error_by_exception_type():
    async with _sse_client("", status_code=500) as client:
        result = await send_request(client, "hi")
    assert result["success"] is False
    assert result["error"].startswith("HTTPStatusError")


@pytest.mark.asyncio
async def test_send_request_passes_sampling_params_in_the_body():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen["body"] = json.loads(request.content)
        return httpx.Response(200, content=b"data: [DONE]\n\n")

    async with load_client("http://test", transport=httpx.MockTransport(handler)) as client:
        params = SamplingParams(max_tokens=16, temperature=0.5, top_p=0.9)
        result = await send_request(client, "hi", sampling_params=params)

    assert result["success"] is True
    assert seen["body"] == {"prompt": "hi", "max_tokens": 16, "temperature": 0.5, "top_p": 0.9}


@pytest.mark.asyncio
async def test_request_sender_sends_its_shapes_in_order_with_each_shape_s_max_tokens(
    monkeypatch,
):
    sent = []

    async def fake_send_request(client, prompt, sampling_params):
        sent.append((prompt, sampling_params))
        return {"success": True}

    monkeypatch.setattr("llm_serving_engine.loadgen.client.send_request", fake_send_request)
    shapes = [
        RequestShape("short", "hi", max_tokens=4, weight=1.0),
        RequestShape("long", "hello", max_tokens=999, weight=1.0),
    ]
    async with load_client("http://test") as client:
        send = request_sender(client, shapes, SamplingParams(temperature=0.5))
        results = [await send() for _ in shapes]

    assert [r["shape"] for r in results] == ["short", "long"]
    assert [(prompt, params.max_tokens, params.temperature) for prompt, params in sent] == [
        ("hi", 4, 0.5),
        ("hello", 999, 0.5),
    ]


def test_shape_schedule_sends_the_asked_for_count_split_by_weight():
    workload = [
        RequestShape("common", "hi", max_tokens=4, weight=0.9),
        RequestShape("rare", "yo", max_tokens=4, weight=0.1),
    ]
    schedule = shape_schedule(workload, 100, random.Random(0), min_per_shape=0)

    counts = Counter(shape.name for shape in schedule)
    assert sum(counts.values()) == 100
    assert counts == {"common": 90, "rare": 10}


def test_shape_schedule_gives_a_rare_shape_a_floor_of_requests():
    workload = [
        RequestShape("common", "hi", max_tokens=4, weight=0.98),
        RequestShape("rare", "yo", max_tokens=4, weight=0.02),
    ]
    schedule = shape_schedule(workload, 100, random.Random(0), min_per_shape=20)

    counts = Counter(shape.name for shape in schedule)
    assert sum(counts.values()) == 100
    assert counts["rare"] >= 20


def test_shape_schedule_keeps_the_workload_s_proportions_when_floors_would_dominate():
    small = Counter(s.name for s in shape_schedule(WORKLOAD, 40, random.Random(0)))
    large = Counter(s.name for s in shape_schedule(WORKLOAD, 400, random.Random(0)))

    assert sum(small.values()) == 40
    # Chat is 75% of the workload's weight; floors may dilute it, but not past half the run.
    assert small["chat"] / 40 > 0.5
    assert large["chunked_document"] >= 20  # the rare shape still reaches its floor


def test_shape_schedule_with_the_same_seed_is_the_same_order():
    def schedule(seed: int) -> list[str]:
        return [shape.name for shape in shape_schedule(WORKLOAD, 60, random.Random(seed))]

    assert schedule(3) == schedule(3)
    assert schedule(3) != schedule(4)


def test_workload_shapes_grow_from_chat_to_a_prompt_past_two_token_budgets():
    by_name = {}
    for shape in WORKLOAD:
        by_name.setdefault(shape.name, []).append(shape)
    longest_chat = max(len(shape.prompt) for shape in by_name["chat"])
    document, long_document = by_name["document"][0], by_name["long_document"][0]
    chunked = by_name["chunked_document"][0]
    assert longest_chat < len(document.prompt) < len(long_document.prompt) < len(chunked.prompt)
    assert by_name["chat"][0].max_tokens < document.max_tokens < long_document.max_tokens
    assert sum(shape.weight for shape in WORKLOAD) == pytest.approx(1.0)


def test_chunked_document_prompt_spans_more_than_two_token_budgets():
    # worker_log lines are 19 Llama-3 tokens each; the prompt must exceed 2 * TOKEN_BUDGET
    # so its prefill takes at least three chunks.
    from llm_serving_engine.scheduling.scheduler import TOKEN_BUDGET

    chunked = next(shape for shape in WORKLOAD if shape.name == "chunked_document")
    assert chunked.prompt.count("\n") * 19 > 2 * TOKEN_BUDGET
