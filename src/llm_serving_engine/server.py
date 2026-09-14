"""HTTP/SSE server shell: FastAPI app factory plus the `llm-serve` entrypoint.

No scheduling or sampling logic lives here — this only tokenizes, hands off to
InferenceEngine.submit, and streams whatever comes back out of output_channels.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from .config import EngineConfig, KVCacheConfig
from .engine import InferenceEngine
from .model.model_runner import ModelRunner
from .model.sampling import SamplingParams
from .model.tokenizer import TokenizerWrapper
from .observability.metrics_export import (
    CONTENT_TYPE_LATEST,
    KV_CACHE_UTILIZATION,
    generate_latest,
    sample_kv_utilization,
)
from .scheduling.dispatch import DONE


class EngineHandle:
    """Owns the currently-loaded InferenceEngine and can replace it wholesale.

    A single GPU has no room to hold two models' weights and KV pools at once, so
    "switching models" means fully loading the replacement, swapping it in, then
    stopping the old engine — never running both.
    """

    def __init__(self, config: EngineConfig):
        self._config = config
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = threading.Lock()
        self._engine = self._load(config.model.model_name_or_path)

    def _load(self, model_name_or_path: str) -> InferenceEngine:
        model_config = replace(self._config.model, model_name_or_path=model_name_or_path)
        tokenizer = TokenizerWrapper(model_name_or_path)
        model_runner = ModelRunner(model_config)
        kv_cache = KVCacheConfig.from_model(model_runner.model.config, model_runner.model.dtype.itemsize)
        self._config = replace(self._config, model=model_config, kv_cache=kv_cache)
        engine = InferenceEngine(self._config, tokenizer, model_runner)
        if self._loop is not None:
            engine.bind_loop(self._loop)
        engine.start()
        return engine

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._engine.bind_loop(loop)

    @property
    def tokenizer(self) -> TokenizerWrapper:
        return self._engine.tokenizer

    def submit(self, prompt: str, sampling_params: SamplingParams) -> tuple[int, asyncio.Queue]:
        return self._engine.submit(prompt, sampling_params)

    def cancel(self, seq_id: int) -> None:
        self._engine.cancel(seq_id)

    @property
    def config(self) -> EngineConfig:
        return self._config

    @property
    def current_model_name(self) -> str:
        return self._config.model.model_name_or_path

    @property
    def allocator(self):
        return self._engine.allocator

    @property
    def running(self):
        return self._engine.running

    def switch_model(self, model_name_or_path: str) -> None:
        """Blocking: drains the current engine, stops it, then loads the replacement.
        Call off the event loop (the /v1/model route routes this through
        asyncio.to_thread) — this is disk I/O plus a GPU weight load.

        Draining before building the new engine matters beyond tidiness: seq_id
        numbering starts at 0 per InferenceEngine instance (engine.py), and
        output_channels (dispatch.py) is one global dict keyed only by seq_id. Flipping
        self._engine while the old one still has in-flight sequences would let a new
        engine's seq_id 0 collide with an old one still writing to the same
        output_channels slot — cross-talk between two unrelated clients' streams. There
        is no such collision once the old engine is confirmed idle before it is retired.
        """
        if model_name_or_path == self.current_model_name:
            return
        with self._lock:
            if model_name_or_path == self.current_model_name:
                return
            old_engine = self._engine
            while old_engine.running or old_engine.waiting:
                time.sleep(0.05)
            old_engine.stop()
            on_cuda = old_engine.model_runner.device.startswith("cuda")
            del old_engine
            if on_cuda:
                import gc

                import torch

                gc.collect()
                torch.cuda.empty_cache()
            self._engine = self._load(model_name_or_path)


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None


class SwitchModelRequest(BaseModel):
    model: str


def _sampling_params(body: GenerateRequest) -> SamplingParams:
    overrides = {
        k: v
        for k, v in (
            ("max_tokens", body.max_tokens),
            ("temperature", body.temperature),
            ("top_p", body.top_p),
        )
        if v is not None
    }
    return SamplingParams(**overrides)


def _format_sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


_KV_SAMPLE_INTERVAL_S = 0.2


async def _sample_kv_utilization_loop(engine) -> None:
    while True:
        utilization = sample_kv_utilization(engine)
        if utilization is not None:
            KV_CACHE_UTILIZATION.set(utilization)
        await asyncio.sleep(_KV_SAMPLE_INTERVAL_S)


def create_app(engine: InferenceEngine) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        engine.bind_loop(asyncio.get_running_loop())
        sampler = asyncio.create_task(_sample_kv_utilization_loop(engine))
        try:
            yield
        finally:
            sampler.cancel()

    app = FastAPI(lifespan=lifespan)
    # Allows the standalone console in web/index.html (opened from file:// or a
    # separate static server) to call this API cross-origin.
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics")
    async def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    if hasattr(engine, "switch_model"):

        @app.get("/v1/model")
        async def get_model() -> dict[str, str]:
            return {"model": engine.current_model_name}

        @app.post("/v1/model")
        async def switch_model(body: SwitchModelRequest) -> dict[str, str]:
            try:
                await asyncio.to_thread(engine.switch_model, body.model)
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e)) from e
            return {"model": engine.current_model_name}

    @app.post("/v1/generate")
    async def generate(body: GenerateRequest) -> StreamingResponse:
        seq_id, output_queue = engine.submit(body.prompt, _sampling_params(body))

        prompt_tokens = len(engine.tokenizer.encode_prompt(body.prompt))

        async def stream() -> AsyncIterator[str]:
            generated: list[int] = []
            try:
                while True:
                    item = await output_queue.get()
                    if item is DONE:
                        yield _format_sse(
                            {"done": True, "prompt_tokens": prompt_tokens, "output_tokens": len(generated)}
                        )
                        break
                    generated.append(item)
                    text = engine.tokenizer.decode_incremental(seq_id, generated)
                    yield _format_sse({"token": text})
            finally:
                engine.cancel(seq_id)
                engine.tokenizer.forget(seq_id)

        return StreamingResponse(stream(), media_type="text/event-stream")

    return app


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    handle = EngineHandle(EngineConfig.from_env())

    import uvicorn

    uvicorn.run(create_app(handle), host=handle.config.server.host, port=handle.config.server.port)


if __name__ == "__main__":
    main()
