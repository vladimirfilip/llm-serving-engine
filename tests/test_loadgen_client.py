import random

import httpx
import pytest

from llm_serving_engine.loadgen.client import (
    WORKLOAD,
    RequestShape,
    load_client,
    request_sender,
    send_request,
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
async def test_request_sender_draws_shapes_by_weight_and_sends_each_shape_s_max_tokens(
    monkeypatch,
):
    sent = []

    async def fake_send_request(client, prompt, sampling_params):
        sent.append((prompt, sampling_params))
        return {"success": True}

    monkeypatch.setattr("llm_serving_engine.loadgen.client.send_request", fake_send_request)
    workload = [
        RequestShape("short", "hi", max_tokens=4, weight=1.0),
        RequestShape("never", "unused", max_tokens=999, weight=0.0),
    ]
    async with load_client("http://test") as client:
        send = request_sender(client, random.Random(0), workload, SamplingParams(temperature=0.5))
        results = [await send() for _ in range(20)]

    assert {r["shape"] for r in results} == {"short"}
    assert {(prompt, params.max_tokens, params.temperature) for prompt, params in sent} == {
        ("hi", 4, 0.5)
    }


@pytest.mark.asyncio
async def test_request_sender_with_the_same_seed_draws_the_same_shapes(monkeypatch):
    async def fake_send_request(client, prompt, sampling_params):
        return {"success": True}

    monkeypatch.setattr("llm_serving_engine.loadgen.client.send_request", fake_send_request)

    async def shapes(seed: int) -> list[str]:
        async with load_client("http://test") as client:
            send = request_sender(client, random.Random(seed))
            return [(await send())["shape"] for _ in range(50)]

    assert await shapes(3) == await shapes(3)
    assert await shapes(3) != await shapes(4)


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
