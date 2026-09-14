import httpx
import pytest

from llm_serving_engine.loadgen.client import DEFAULT_PROMPTS, LoadGenConfig, send_request
from llm_serving_engine.model.sampling import SamplingParams


def _sse_transport(body: str, status_code: int = 200) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, content=body.encode(), headers={"content-type": "text/event-stream"})

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_send_request_counts_tokens_and_measures_first_token_latency():
    body = (
        'data: {"token": "Hello"}\n\ndata: {"token": " world"}\n\n'
        'data: {"done": true, "prompt_tokens": 5, "output_tokens": 2}\n\n'
    )
    result = await send_request(
        "http://test", "hi", transport=_sse_transport(body)
    )
    assert result["success"] is True
    assert result["error"] is None
    assert result["num_tokens_received"] == 2
    assert result["first_token_latency"] is not None
    assert result["first_token_latency"] >= 0
    assert "latency" not in result  # only the timing loop measures latency
    assert result["prompt_tokens"] == 5
    assert result["output_tokens"] == 2
    assert len(result["token_times"]) == 2
    assert result["token_times"] == sorted(result["token_times"])


@pytest.mark.asyncio
async def test_send_request_reports_stream_level_error():
    body = 'data: {"error": "generation failed"}\n\n'
    result = await send_request("http://test", "hi", transport=_sse_transport(body))
    assert result["success"] is False
    assert result["error"] == "generation failed"


@pytest.mark.asyncio
async def test_send_request_reports_http_error_status():
    result = await send_request("http://test", "hi", transport=_sse_transport("", status_code=500))
    assert result["success"] is False
    assert result["error"] is not None
    assert result["num_tokens_received"] == 0


@pytest.mark.asyncio
async def test_send_request_passes_sampling_params_in_body():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen["body"] = json.loads(request.content)
        return httpx.Response(200, content=b"data: [DONE]\n\n")

    result = await send_request(
        "http://test",
        "hi",
        sampling_params=SamplingParams(max_tokens=16, temperature=0.5, top_p=0.9),
        transport=httpx.MockTransport(handler),
    )
    assert result["success"] is True
    assert seen["body"] == {"prompt": "hi", "max_tokens": 16, "temperature": 0.5, "top_p": 0.9}


def test_load_gen_config_samples_from_default_prompts():
    config = LoadGenConfig(target_qps=1.0, duration_s=1.0)
    assert config.sample_prompt() in DEFAULT_PROMPTS


def test_load_gen_config_samples_from_custom_prompts():
    config = LoadGenConfig(target_qps=1.0, duration_s=1.0, prompts=["only-one"])
    assert config.sample_prompt() == "only-one"
