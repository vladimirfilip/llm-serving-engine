from __future__ import annotations

import asyncio
import json

import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from llm_serving_engine.dispatch import DONE, output_channels  # noqa: E402
from llm_serving_engine.sampling import SamplingParams  # noqa: E402
from llm_serving_engine.server import create_app  # noqa: E402


class FakeTokenizer:
    def encode_prompt(self, prompt: str) -> list[int]:
        return list(range(len(prompt.split())))

    def decode_incremental(self, seq_id: int, generated_tokens: list[int]) -> str:
        return f"<{generated_tokens[-1]}>"

    def forget(self, seq_id: int) -> None:
        pass


class FakeEngine:
    """Stub standing in for InferenceEngine: no model, no scheduler thread."""

    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.cancelled: list[int] = []
        self.submitted: list[tuple[str, SamplingParams]] = []
        self._next_id = 0

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        pass

    def submit(self, prompt: str, sampling_params: SamplingParams):
        seq_id = self._next_id
        self._next_id += 1
        self.submitted.append((prompt, sampling_params))
        q: asyncio.Queue = asyncio.Queue(maxsize=64)
        output_channels[seq_id] = q
        return seq_id, q

    def cancel(self, seq_id: int) -> None:
        self.cancelled.append(seq_id)
        output_channels.pop(seq_id, None)


@pytest.fixture
def engine():
    return FakeEngine()


@pytest.fixture
def client(engine):
    app = create_app(engine)
    with TestClient(app) as c:
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_metrics_exposes_prometheus_text(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "text/plain" in resp.headers["content-type"]


def _sse_events(raw_text: str) -> list[dict]:
    events = []
    for block in raw_text.strip().split("\n\n"):
        if block.startswith("data: "):
            events.append(json.loads(block[len("data: ") :]))
    return events


def test_generate_streams_tokens_then_terminal_event(client, engine):
    seq_id_holder = {}
    real_submit = engine.submit

    def submit_and_capture(prompt, sampling_params):
        seq_id, q = real_submit(prompt, sampling_params)
        seq_id_holder["seq_id"] = seq_id
        for tok in (11, 22, 33):
            q.put_nowait(tok)
        q.put_nowait(DONE)
        return seq_id, q

    engine.submit = submit_and_capture

    resp = client.post("/v1/generate", json={"prompt": "hello", "max_tokens": 3})
    assert resp.status_code == 200
    events = _sse_events(resp.text)

    assert events[:-1] == [{"token": "<11>"}, {"token": "<22>"}, {"token": "<33>"}]
    assert events[-1] == {"done": True, "prompt_tokens": 1, "output_tokens": 3}

    seq_id = seq_id_holder["seq_id"]
    assert seq_id in engine.cancelled  # finally always cancels, DONE or not
    assert seq_id not in output_channels


def test_generate_request_builds_sampling_params(client, engine):
    def submit_and_finish(prompt, sampling_params):
        seq_id, q = FakeEngine.submit(engine, prompt, sampling_params)
        q.put_nowait(DONE)  # no tokens needed; this test only checks what was submitted
        return seq_id, q

    engine.submit = submit_and_finish

    client.post(
        "/v1/generate",
        json={"prompt": "hi", "max_tokens": 10, "temperature": 0.5, "top_p": 0.9},
    )
    prompt, params = engine.submitted[0]
    assert prompt == "hi"
    assert params.max_tokens == 10
    assert params.temperature == 0.5
    assert params.top_p == 0.9


def test_generate_defaults_when_fields_omitted(client, engine):
    def submit_and_finish(prompt, sampling_params):
        seq_id, q = FakeEngine.submit(engine, prompt, sampling_params)
        q.put_nowait(DONE)
        return seq_id, q

    engine.submit = submit_and_finish

    client.post("/v1/generate", json={"prompt": "hi"})
    _, params = engine.submitted[0]
    assert params == SamplingParams()


@pytest.mark.asyncio
async def test_loadgen_client_parses_this_server_s_sse_stream():
    """The two halves of the SSE contract meet only here: server.py writes the events and
    loadgen/client.py parses them, so nothing else catches the two drifting apart."""
    from llm_serving_engine.loadgen.client import send_request

    engine = FakeEngine()
    app = create_app(engine)
    real_submit = engine.submit

    def submit_and_finish(prompt, sampling_params):
        seq_id, q = real_submit(prompt, sampling_params)
        for tok in (1, 2, 3):
            q.put_nowait(tok)
        q.put_nowait(DONE)
        return seq_id, q

    engine.submit = submit_and_finish

    result = await send_request(
        "http://test", "hello", transport=httpx.ASGITransport(app=app)
    )
    assert result["success"] is True
    assert result["num_tokens_received"] == 3
    assert result["first_token_latency"] is not None
    assert result["prompt_tokens"] == 1
    assert result["output_tokens"] == 3
    assert len(result["token_times"]) == 3


@pytest.mark.asyncio
async def test_disconnect_cleans_up_output_channel():
    """A client that stops reading mid-stream must not leave a stale output_channels entry.

    Starlette's synchronous TestClient always drains an ASGI call to completion, so it can't
    model a client that walks away mid-stream. httpx.AsyncClient over ASGITransport runs the
    app as a real awaitable, so cancelling that await — what a dropped connection ultimately
    causes on a live server — is the faithful way to exercise this path.
    """
    engine = FakeEngine()
    app = create_app(engine)
    real_submit = engine.submit

    def submit_with_one_token(prompt, sampling_params):
        seq_id, q = real_submit(prompt, sampling_params)
        q.put_nowait(42)  # one chunk, then the queue is starved forever — no DONE
        return seq_id, q

    engine.submit = submit_with_one_token

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        with pytest.raises(asyncio.TimeoutError):
            post = client.post("/v1/generate", json={"prompt": "hello"})
            await asyncio.wait_for(post, timeout=0.2)

    seq_id = engine._next_id - 1
    assert seq_id not in output_channels
    assert seq_id in engine.cancelled
