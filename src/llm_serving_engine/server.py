"""HTTP/SSE server: the FastAPI app factory and the `llm-serve` entrypoint. Requests go
through `submit`; tokens stream back out of `output_channels`."""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import AsyncIterator

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from .config import EngineConfig, KVCacheConfig
from .engine import EngineUnavailable, InferenceEngine, Submission
from .model.model_runner import ModelRunner
from .model.sampling import SamplingParams
from .model.tokenizer import TokenizerWrapper
from .observability.metrics_export import CONTENT_TYPE_LATEST, KV_CACHE_UTILIZATION, generate_latest
from .scheduling.dispatch import ABORTED, DONE, output_channels

logger = logging.getLogger(__name__)

_KV_SAMPLE_INTERVAL_S = 0.2
_DRAIN_POLL_S = 0.05


class EngineHandle:
    """Owns the loaded InferenceEngine and replaces it wholesale on a model switch.

    One GPU can't hold two models' weights and KV pools, so a switch drains and stops the
    old engine, releases its memory, then loads the replacement. Requests arriving in
    between get EngineUnavailable.
    """

    def __init__(self, config: EngineConfig):
        self._config = config
        self._loop: asyncio.AbstractEventLoop | None = None
        self._switch_lock = threading.Lock()
        self._engine: InferenceEngine | None = self._load(config.model.model_name_or_path)

    def _load(self, model_name_or_path: str) -> InferenceEngine:
        model_config = replace(self._config.model, model_name_or_path=model_name_or_path)
        tokenizer = TokenizerWrapper(model_name_or_path)
        model_runner = ModelRunner(model_config)
        kv_cache = KVCacheConfig.from_model(
            model_runner.model.config, model_runner.model.dtype.itemsize
        )
        config = replace(self._config, model=model_config, kv_cache=kv_cache)
        engine = InferenceEngine(config, tokenizer, model_runner)
        if self._loop is not None:
            engine.bind_loop(self._loop)
        engine.start()
        self._config = config
        return engine

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        if self._engine is not None:
            self._engine.bind_loop(loop)

    def submit(self, prompt: str, sampling_params: SamplingParams) -> Submission:
        engine = self._engine
        if engine is None:
            raise EngineUnavailable("switching models")
        return engine.submit(prompt, sampling_params)

    @property
    def config(self) -> EngineConfig:
        return self._config

    @property
    def current_model_name(self) -> str:
        return self._config.model.model_name_or_path

    @property
    def healthy(self) -> bool:
        return self._engine is not None and self._engine.healthy

    @property
    def kv_utilization(self) -> float | None:
        engine = self._engine
        return engine.kv_utilization if engine is not None else None

    def switch_model(self, model_name_or_path: str) -> None:
        """Blocking disk and GPU work: call off the event loop. If the new model fails to
        load, its memory is released, the previous model reloads, and the error is
        re-raised. If that reload fails too, the handle serves nothing until a later
        switch succeeds."""
        with self._switch_lock:
            if model_name_or_path == self.current_model_name and self._engine is not None:
                return
            previous = self.current_model_name
            old_engine, self._engine = self._engine, None
            if old_engine is not None:
                old_engine.close_ingress()
                while not old_engine.is_idle:
                    time.sleep(_DRAIN_POLL_S)
                old_engine.stop()
            del old_engine
            _release_gpu_memory()

            try:
                self._engine = self._load(model_name_or_path)
                return
            except Exception as e:
                logger.exception("loading %s failed; reloading %s", model_name_or_path, previous)
                # The traceback's frames hold the failed model's tensors.
                error = e.with_traceback(None)
            _release_gpu_memory()
            self._engine = self._load(previous)
            raise error


def _release_gpu_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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


async def _sample_kv_utilization_loop(engine: InferenceEngine | EngineHandle) -> None:
    while True:
        utilization = engine.kv_utilization
        if utilization is not None:
            KV_CACHE_UTILIZATION.set(utilization)
        await asyncio.sleep(_KV_SAMPLE_INTERVAL_S)


def create_app(engine: InferenceEngine | EngineHandle) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        engine.bind_loop(asyncio.get_running_loop())
        sampler = asyncio.create_task(_sample_kv_utilization_loop(engine))
        try:
            yield
        finally:
            sampler.cancel()

    app = FastAPI(lifespan=lifespan)
    # Lets a browser console served from another origin (or file://) call this API.
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
    )

    @app.get("/health")
    async def health() -> JSONResponse:
        if engine.healthy:
            return JSONResponse({"status": "ok"})
        return JSONResponse({"status": "unavailable"}, status_code=503)

    @app.get("/metrics")
    async def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    if isinstance(engine, EngineHandle):

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
        try:
            submission = engine.submit(body.prompt, _sampling_params(body))
        except EngineUnavailable as e:
            raise HTTPException(status_code=503, detail=str(e)) from e

        async def stream() -> AsyncIterator[str]:
            seq_id, tokenizer = submission.seq_id, submission.tokenizer
            generated: list[int] = []
            try:
                while True:
                    item = await submission.output_queue.get()
                    if item is DONE:
                        yield _format_sse(
                            {
                                "done": True,
                                "prompt_tokens": submission.prompt_len,
                                "output_tokens": len(generated),
                            }
                        )
                        break
                    if item is ABORTED:
                        yield _format_sse({"error": "request aborted by the engine"})
                        break
                    generated.append(item)
                    # One event per token, even while its text is held back, so clients
                    # timing token events measure every token.
                    yield _format_sse({"token": tokenizer.decode_incremental(seq_id, generated)})
            finally:
                output_channels.pop(seq_id, None)
                tokenizer.forget(seq_id)

        return StreamingResponse(stream(), media_type="text/event-stream")

    return app


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    handle = EngineHandle(EngineConfig.from_env())

    import uvicorn

    uvicorn.run(create_app(handle), host=handle.config.server.host, port=handle.config.server.port)


if __name__ == "__main__":
    main()
