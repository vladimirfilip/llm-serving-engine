from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import replace

import httpx
import pytest
from fastapi.testclient import TestClient

from llm_serving_engine.engine import (
    EngineUnavailable,
    InferenceEngine,
    InvalidPrompt,
    Submission,
)
from llm_serving_engine.model.sampling import SamplingParams
from llm_serving_engine.scheduling.dispatch import (
    ABORTED,
    DONE,
    new_output_channel,
    output_channels,
)
from llm_serving_engine.server import EngineHandle, create_app
from tests.factories import TOKEN, FakeModelRunner, FakeTokenizer, make_config, read_stream


class FakeStreamTokenizer:
    def decode_incremental(self, seq_id: int, generated_tokens: list[int]) -> str:
        return f"<{generated_tokens[-1]}>"

    def forget(self, seq_id: int) -> None:
        pass


class FakeEngine:
    """Stands in for InferenceEngine: `on_submit` fills each new output queue."""

    def __init__(self, on_submit=lambda q: q.put_nowait(DONE)):
        self.on_submit = on_submit
        self.submitted: list[tuple[str, SamplingParams]] = []
        self.healthy = True
        self.accepting = True
        self.kv_utilization = 0.0

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        pass

    def submit(self, prompt: str, sampling_params: SamplingParams) -> Submission:
        if not self.accepting:
            raise EngineUnavailable("closed")
        if not prompt:
            raise InvalidPrompt("empty")
        self.submitted.append((prompt, sampling_params))
        seq_id, q = new_output_channel(maxsize=64)
        self.on_submit(q)
        return Submission(seq_id, q, len(prompt.split()), FakeStreamTokenizer())


def put_all(*items):
    def fill(q: asyncio.Queue) -> None:
        for item in items:
            q.put_nowait(item)

    return fill


def sse_events(raw_text: str) -> list[dict]:
    return [
        json.loads(block[len("data: ") :])
        for block in raw_text.strip().split("\n\n")
        if block.startswith("data: ")
    ]


def test_health_reports_ok_while_the_engine_is_healthy():
    engine = FakeEngine()
    with TestClient(create_app(engine)) as client:
        assert client.get("/health").json() == {"status": "ok"}
        engine.healthy = False
        assert client.get("/health").status_code == 503


def test_metrics_exposes_prometheus_text():
    with TestClient(create_app(FakeEngine())) as client:
        resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "text/plain" in resp.headers["content-type"]


def test_generate_streams_tokens_then_a_done_event_and_releases_the_channel():
    engine = FakeEngine(on_submit=put_all(11, 22, 33, DONE))
    with TestClient(create_app(engine)) as client:
        resp = client.post("/v1/generate", json={"prompt": "hello", "max_tokens": 3})

    assert sse_events(resp.text) == [
        {"token": "<11>"},
        {"token": "<22>"},
        {"token": "<33>"},
        {"done": True, "prompt_tokens": 1, "output_tokens": 3},
    ]
    assert output_channels == {}


def test_an_aborted_request_ends_with_an_error_event():
    engine = FakeEngine(on_submit=put_all(11, ABORTED))
    with TestClient(create_app(engine)) as client:
        resp = client.post("/v1/generate", json={"prompt": "hello"})

    events = sse_events(resp.text)
    assert events[0] == {"token": "<11>"}
    assert "error" in events[-1]
    assert output_channels == {}


def test_generate_is_503_while_the_engine_accepts_nothing():
    engine = FakeEngine()
    engine.accepting = False
    with TestClient(create_app(engine)) as client:
        assert client.post("/v1/generate", json={"prompt": "hi"}).status_code == 503


def test_generate_is_400_for_a_prompt_the_model_cannot_run():
    with TestClient(create_app(FakeEngine())) as client:
        assert client.post("/v1/generate", json={"prompt": ""}).status_code == 400


def test_generate_builds_sampling_params_from_the_body():
    engine = FakeEngine()
    with TestClient(create_app(engine)) as client:
        body = {"prompt": "hi", "max_tokens": 10, "temperature": 0.5, "top_p": 0.9}
        client.post("/v1/generate", json=body)
        client.post("/v1/generate", json={"prompt": "hi"})

    (prompt, params), (_, defaults) = engine.submitted
    assert prompt == "hi"
    assert (params.max_tokens, params.temperature, params.top_p) == (10, 0.5, 0.9)
    assert defaults == SamplingParams()


@pytest.mark.asyncio
async def test_a_client_that_disconnects_mid_stream_releases_its_channel():
    """TestClient drains every response, so it can't walk away mid-stream; cancelling an
    httpx request over ASGITransport is what a dropped connection does to the app."""
    app = create_app(FakeEngine(on_submit=put_all(42)))  # one token, then silence

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(client.post("/v1/generate", json={"prompt": "hello"}), 0.2)

    assert output_channels == {}


class GatedModelRunner(FakeModelRunner):
    """Holds every forward pass until `gate` is set."""

    def __init__(self, gate: threading.Event):
        super().__init__()
        self.gate = gate

    def forward(self, plan, seqs):
        self.gate.wait()
        return super().forward(plan, seqs)


@pytest.fixture
def handle_factory(monkeypatch):
    """EngineHandle whose _load builds a fake-runner engine; names in `broken` fail."""
    broken = {"broken"}
    gate = threading.Event()
    gate.set()

    def fake_load(self, model_name_or_path):
        if model_name_or_path in broken:
            raise OSError("no such model")
        config = replace(
            self._config, model=replace(self._config.model, model_name_or_path=model_name_or_path)
        )
        engine = InferenceEngine(config, FakeTokenizer(), GatedModelRunner(gate))
        if self._loop is not None:
            engine.bind_loop(self._loop)
        engine.start()
        self._config = config
        return engine

    monkeypatch.setattr(EngineHandle, "_load", fake_load)
    handles = []

    def make() -> EngineHandle:
        handle = EngineHandle(make_config())
        handles.append(handle)
        return handle

    yield make, gate, broken
    gate.set()
    for handle in handles:
        if handle._engine is not None:
            handle._engine.stop()


@pytest.mark.asyncio
async def test_switch_model_drains_in_flight_requests_and_rejects_new_ones_meanwhile(
    handle_factory,
):
    make, gate, _broken = handle_factory
    handle = make()
    handle.bind_loop(asyncio.get_running_loop())
    gate.clear()
    in_flight = handle.submit("hi", SamplingParams(max_tokens=1))

    switch = asyncio.create_task(asyncio.to_thread(handle.switch_model, "other"))
    while True:
        try:
            late = handle.submit("hi", SamplingParams(max_tokens=1))
        except EngineUnavailable:
            break
        output_channels.pop(late.seq_id)
        await asyncio.sleep(0.01)
    assert not switch.done()  # still waiting on the gated in-flight request

    gate.set()
    assert await read_stream(in_flight.output_queue) == [TOKEN, DONE]
    await asyncio.wait_for(switch, timeout=5)

    assert handle.current_model_name == "other"
    after = handle.submit("hi", SamplingParams(max_tokens=1))
    assert after.seq_id != in_flight.seq_id
    assert await read_stream(after.output_queue) == [TOKEN, DONE]


def test_a_failed_switch_reloads_the_previous_model(handle_factory):
    make, _gate, _broken = handle_factory
    handle = make()
    previous = handle.current_model_name

    with pytest.raises(OSError):
        handle.switch_model("broken")

    assert handle.current_model_name == previous
    assert handle.healthy


def test_a_handle_whose_fallback_also_failed_recovers_on_the_next_switch(handle_factory):
    make, _gate, broken = handle_factory
    handle = make()
    broken.add(handle.current_model_name)

    with pytest.raises(OSError):
        handle.switch_model("broken")
    assert not handle.healthy

    handle.switch_model("other")
    assert handle.current_model_name == "other"
    assert handle.healthy


def test_model_routes_exist_only_for_a_switchable_engine(handle_factory):
    make, _gate, _broken = handle_factory
    with TestClient(create_app(make())) as client:
        assert client.get("/v1/model").json() == {"model": make_config().model.model_name_or_path}
    with TestClient(create_app(FakeEngine())) as client:
        assert client.get("/v1/model").status_code == 404
