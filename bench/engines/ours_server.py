"""An OpenAI-style `/v1/completions` server around the engine, so the harness drives ours the
way it drives every baseline. Kept thin: this wrapper is part of what gets measured.
`python -m bench.engines.ours_server --port N --max-model-len L`, the model from `LLM_MODEL`."""

from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator

import orjson
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from llm_serving_engine.config import EngineConfig
from llm_serving_engine.engine import EngineUnavailable, InvalidPrompt, Submission
from llm_serving_engine.model.sampling import SamplingParams
from llm_serving_engine.scheduling.dispatch import ABORTED, DONE, output_channels
from llm_serving_engine.server import EngineHandle

SSE_HEADERS = {"Cache-Control": "no-cache"}


def sampling_params(body: dict) -> SamplingParams:
    return SamplingParams(
        temperature=body.get("temperature", 1.0), top_p=body.get("top_p", 1.0),
        max_tokens=body["max_tokens"], ignore_eos=bool(body.get("ignore_eos", False)),
    )


def event(data: dict) -> bytes:
    return b"data: " + orjson.dumps(data) + b"\n\n"


def token_event(text: str, token: int | None, logprob: float | None) -> bytes:
    choice: dict = {"index": 0, "text": text, "finish_reason": None}
    if logprob is not None:
        choice["logprobs"] = {"tokens": [f"token_id:{token}"], "token_logprobs": [logprob]}
    return event({"choices": [choice]})


def usage_of(submission: Submission, generated: int) -> dict:
    prompt = submission.prompt_len
    return {"prompt_tokens": prompt, "completion_tokens": generated,
            "total_tokens": prompt + generated}


async def stream_tokens(
    submission: Submission, include_usage: bool, skip_special_tokens: bool
) -> AsyncIterator[bytes]:
    """One SSE event per generated token, its text empty while a multi-byte character is held
    back. The finally pops this stream's channel whether or not the client stayed."""
    seq_id, tokenizer = submission.seq_id, submission.tokenizer
    generated: list[int] = []
    try:
        while True:
            item = await submission.output_queue.get()
            if item is DONE:
                break
            if item is ABORTED:
                yield event({"error": "request aborted by the engine"})
                return
            generated.append(item)
            text = tokenizer.decode_incremental(seq_id, generated, skip_special_tokens)
            logprobs = submission.logprobs
            logprob = logprobs[len(generated) - 1] if logprobs is not None else None
            yield token_event(text, item, logprob)
        if submission.logprobs is not None and len(submission.logprobs) != len(generated):
            # the bounded channel dropped tokens, so positional logprobs would be misaligned
            yield event({"error": f"{len(submission.logprobs) - len(generated)} token(s) "
                                  f"dropped from a logprob stream"})
            return
        if include_usage:
            yield event({"choices": [], "usage": usage_of(submission, len(generated))})
        yield b"data: [DONE]\n\n"
    finally:
        output_channels.pop(seq_id, None)
        tokenizer.forget(seq_id)


def cut_at_stop(text: str, stops: list[str]) -> tuple[str, bool]:
    """`text` up to the earliest stop sequence, and whether one was found."""
    cuts = [i for stop in stops if (i := text.find(stop)) >= 0]
    return (text[: min(cuts)], True) if cuts else (text, False)


async def collect_tokens(submission: Submission, stops: list[str]) -> tuple[list[int], bool]:
    """Every token of a non-streamed request up to its stop sequence, and whether it finished
    rather than aborted. This wrapper doesn't call `submission.cancel` on a stop sequence, so
    the request keeps generating past it and its remaining tokens are dropped here."""
    seq_id = submission.seq_id
    generated: list[int] = []
    try:
        while (item := await submission.output_queue.get()) is not DONE:
            if item is ABORTED:
                return generated, False
            generated.append(item)
            if stops and cut_at_stop(submission.tokenizer.decode(generated), stops)[1]:
                break
        return generated, True
    finally:
        output_channels.pop(seq_id, None)
        submission.tokenizer.forget(seq_id)


def create_app(handle: EngineHandle, max_model_len: int) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        handle.bind_loop(asyncio.get_running_loop())
        yield

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse({"status": "ok"}, status_code=200 if handle.healthy else 503)

    @app.get("/internal/stats")
    async def stats() -> dict:
        return handle.engine.stats()

    @app.post("/internal/score")
    async def score(request: Request) -> Response:
        token_ids = (await request.json())["token_ids"]
        logprobs = await asyncio.to_thread(handle.engine.model_runner.score, token_ids)
        return Response(orjson.dumps({"logprobs": logprobs}), media_type="application/json")

    @app.post("/v1/completions")
    async def completions(request: Request) -> Response:
        body = orjson.loads(await request.body())
        prompt = body["prompt"]
        tokens = handle.engine.tokenizer.encode(prompt) if isinstance(prompt, str) else prompt
        if len(tokens) + body["max_tokens"] > max_model_len:
            raise HTTPException(
                400, f"prompt plus max_tokens exceeds max_model_len {max_model_len}"
            )
        try:
            submission = handle.submit_tokens(tokens, sampling_params(body),
                                              bool(body.get("logprobs")))
        except EngineUnavailable as e:
            raise HTTPException(503, str(e)) from e
        except InvalidPrompt as e:
            raise HTTPException(400, str(e)) from e
        if body.get("stream"):
            include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
            skip_special = bool(body.get("skip_special_tokens", True))
            return StreamingResponse(stream_tokens(submission, include_usage, skip_special),
                                     media_type="text/event-stream", headers=SSE_HEADERS)
        stops = [body["stop"]] if isinstance(body.get("stop"), str) else body.get("stop") or []
        generated, finished = await collect_tokens(submission, stops)
        if not finished:
            raise HTTPException(500, "request aborted by the engine")
        text = cut_at_stop(handle.engine.tokenizer.decode(generated), stops)[0]
        return Response(orjson.dumps({"choices": [{"index": 0, "text": text}],
                                      "usage": usage_of(submission, len(generated))}),
                        media_type="application/json")

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--max-model-len", type=int, required=True)
    args = parser.parse_args()

    import uvicorn

    handle = EngineHandle(EngineConfig.from_env())
    uvicorn.run(create_app(handle, args.max_model_len), host="127.0.0.1", port=args.port,
                loop="uvloop", log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
