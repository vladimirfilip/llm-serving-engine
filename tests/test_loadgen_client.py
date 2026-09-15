import httpx
import pytest

from llm_serving_engine.loadgen.client import (
    DEFAULT_PROMPTS,
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
async def test_send_request_counts_tokens_and_measures_first_token_latency():
    body = (
        'data: {"token": "Hello"}\n\ndata: {"token": " world"}\n\n'
        'data: {"done": true, "prompt_tokens": 5, "output_tokens": 2}\n\n'
    )
    async with _sse_client(body) as client:
        result = await send_request(client, "hi")

    assert result["success"] is True
    assert result["error"] is None
    assert result["num_tokens_received"] == 2
    assert result["first_token_latency"] >= 0
    assert "latency" not in result  # only the timing loop measures latency
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
async def test_request_sender_sends_one_of_its_prompts(monkeypatch):
    sent = []

    async def fake_send_request(client, prompt, sampling_params):
        sent.append(prompt)
        return {"success": True}

    monkeypatch.setattr("llm_serving_engine.loadgen.client.send_request", fake_send_request)
    async with load_client("http://test") as client:
        await request_sender(client)()
        await request_sender(client, prompts=["only-one"])()

    assert sent[0] in DEFAULT_PROMPTS
    assert sent[1] == "only-one"
