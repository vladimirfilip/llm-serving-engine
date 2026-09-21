"""A fake engine for tests and CPU-only runs: OpenAI-style streaming completions, `/health`,
`/internal/stats` and Prometheus `/metrics`, with latencies from a simple continuous-batching
model. `python -m bench.engines.mock_server --port N [--knob value ...]`."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
import time
from collections import deque
from dataclasses import dataclass, fields

import orjson
from aiohttp import web

VOCAB = 128256


@dataclass(slots=True)
class MockConfig:
    prefill_ms_per_1k: float = 50.0
    decode_ms: float = 20.0  # base cost of one decode step
    decode_ms_per_seq: float = 0.0  # added per running sequence
    max_num_seqs: int = 64
    chunk_tokens: int = 0  # prefill tokens per step alongside decode; 0 prefills in one block
    prefix_cache: bool = False
    burst: int = 1  # tokens released together per stream event group
    bundle_events: int = 1  # tokens packed into each streamed event
    kv_blocks: int = 4096
    block_size: int = 16
    crash_running: int = 0  # exit the process once this many sequences run together; 0 never
    token_ids: bool = True  # name each logprob token by id; False gives plain text, like SGLang


@dataclass(slots=True)
class Stream:
    prompt: list[int]
    max_tokens: int
    logprobs: bool
    queue: asyncio.Queue
    prefill_left: float  # ms of prefill work still to do
    generated: int = 0
    prefilled: bool = False


def token_at(prompt: list[int], index: int) -> int:
    """Deterministic pseudo-token: a function of the prompt and position only, so the result
    does not depend on batching."""
    key = f"{sum(prompt)}:{len(prompt)}:{index}".encode()
    digest = hashlib.blake2b(key, digest_size=4).digest()
    return int.from_bytes(digest, "little") % VOCAB


def logprob_at(prompt: list[int], index: int) -> float:
    return -(token_at(prompt, index) % 1000) / 1000


class MockEngine:
    def __init__(self, cfg: MockConfig):
        self.cfg = cfg
        self.waiting: deque[Stream] = deque()
        self.running: list[Stream] = []
        self.seen_prompts: set[tuple[int, ...]] = set()
        self.wake = asyncio.Event()
        self.preemptions = 0

    def submit(self, prompt: list[int], max_tokens: int, logprobs: bool) -> Stream:
        cached = self.cfg.prefix_cache and tuple(prompt) in self.seen_prompts
        self.seen_prompts.add(tuple(prompt))
        prefill = 0.0 if cached else self.cfg.prefill_ms_per_1k * len(prompt) / 1000
        stream = Stream(prompt, max_tokens, logprobs, asyncio.Queue(), prefill)
        self.waiting.append(stream)
        self.wake.set()
        return stream

    def _emit(self, stream: Stream, tokens: int) -> None:
        for _ in range(tokens):
            i = stream.generated
            stream.generated += 1
            done = stream.generated >= stream.max_tokens
            item = (token_at(stream.prompt, i), logprob_at(stream.prompt, i), done)
            stream.queue.put_nowait(item)
            if done:
                return

    async def run(self) -> None:
        cfg = self.cfg
        deadline = time.perf_counter()
        while True:
            while self.waiting and len(self.running) < cfg.max_num_seqs:
                self.running.append(self.waiting.popleft())
            if not self.running:
                self.wake.clear()
                await self.wake.wait()
                deadline = time.perf_counter()
                continue
            if 0 < cfg.crash_running <= len(self.running):
                os._exit(1)
            decoding = [s for s in self.running if s.prefilled]
            step_ms, prefill_budget = 0.0, cfg.chunk_tokens
            just_prefilled = []
            for stream in self.running:
                if stream.prefilled:
                    continue
                if cfg.chunk_tokens == 0:
                    step_ms += stream.prefill_left
                    stream.prefill_left = 0.0
                elif prefill_budget > 0 and cfg.prefill_ms_per_1k > 0:
                    work = min(stream.prefill_left, cfg.prefill_ms_per_1k * prefill_budget / 1000)
                    step_ms += work
                    stream.prefill_left -= work
                    prefill_budget -= round(work * 1000 / cfg.prefill_ms_per_1k)
                stream.prefilled = stream.prefill_left <= 1e-9
                if stream.prefilled:
                    just_prefilled.append(stream)
            if decoding:
                step_ms += cfg.decode_ms + cfg.decode_ms_per_seq * len(decoding)
            # A step takes at least its own duration: a timer that overshot is not made up by
            # running the next steps early, which would release tokens in bursts.
            deadline = max(deadline, time.perf_counter()) + step_ms / 1000
            await asyncio.sleep(max(0.0, deadline - time.perf_counter()))
            for stream in decoding:
                self._emit(stream, cfg.burst)
            for stream in just_prefilled:
                self._emit(stream, 1)
            self.running = [s for s in self.running if s.generated < s.max_tokens]

    def stats(self) -> dict:
        used_tokens = sum(len(s.prompt) + s.generated for s in self.running)
        blocks_used = -(-used_tokens // self.cfg.block_size)
        return {
            "running": len(self.running),
            "waiting": len(self.waiting),
            "preemptions_total": self.preemptions,
            "kv": {"block_size": self.cfg.block_size, "blocks_total": self.cfg.kv_blocks,
                   "blocks_used": blocks_used, "tokens_used": used_tokens},
            "memory_bytes": {"weights": 0, "kv_cache": 0, "activations": 0, "workspace": 0,
                             "cuda_graph_pool": 0, "other": 0},
        }


def chunk(token: int, logprob: float | None, ids: bool = True) -> bytes:
    choice = {"index": 0, "text": f" t{token}", "finish_reason": None}
    if logprob is not None:
        name = f"token_id:{token}" if ids else f" t{token}"
        choice["logprobs"] = {"tokens": [name], "token_logprobs": [logprob]}
    return b"data: " + orjson.dumps({"choices": [choice]}) + b"\n\n"


async def completions(request: web.Request) -> web.StreamResponse:
    engine: MockEngine = request.app["engine"]
    body = await request.json()
    prompt = body["prompt"]
    if isinstance(prompt, str):
        prompt = [ord(c) % VOCAB for c in prompt]
    stream = engine.submit(prompt, body["max_tokens"], bool(body.get("logprobs")))
    if not body.get("stream"):
        tokens = [(await stream.queue.get())[0] for _ in range(stream.max_tokens)]
        usage = {"prompt_tokens": len(prompt), "completion_tokens": len(tokens)}
        return web.json_response({"choices": [{"text": "".join(f" t{t}" for t in tokens)}],
                                  "usage": usage})
    response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await response.prepare(request)
    sent, bundle = 0, engine.cfg.bundle_events
    pending: list[int] = []
    while sent < stream.max_tokens:
        token, logprob, _done = await stream.queue.get()
        sent += 1
        pending.append(token)
        if bundle == 1:
            await response.write(chunk(token, logprob if stream.logprobs else None,
                                       engine.cfg.token_ids))
        elif len(pending) == bundle or sent == stream.max_tokens:
            text = "".join(f" t{t}" for t in pending)
            await response.write(b"data: " + orjson.dumps(
                {"choices": [{"index": 0, "text": text, "finish_reason": None}]}) + b"\n\n")
            pending.clear()
    usage = {"prompt_tokens": len(prompt), "completion_tokens": sent,
             "total_tokens": len(prompt) + sent}
    await response.write(b"data: " + orjson.dumps({"choices": [], "usage": usage}) + b"\n\n")
    await response.write(b"data: [DONE]\n\n")
    return response


async def score(request: web.Request) -> web.Response:
    """Per-token logprobs of a token sequence, `len(ids) - 1` of them."""
    ids = (await request.json())["token_ids"]
    return web.json_response({"logprobs": [logprob_at(ids[:i], i) for i in range(1, len(ids))]})


def metrics_text(engine: MockEngine) -> str:
    s = engine.stats()
    return (
        f"mock:num_requests_running {s['running']}\nmock:num_requests_waiting {s['waiting']}\n"
        f"mock:kv_cache_usage_perc {s['kv']['blocks_used'] / s['kv']['blocks_total']}\n"
        f"mock:num_preemptions_total {s['preemptions_total']}\n"
    )


def parse_bool(text: str) -> bool:
    return text.lower() in ("1", "true", "yes")


def make_app(cfg: MockConfig) -> web.Application:
    app = web.Application()
    engine = MockEngine(cfg)
    app["engine"] = engine

    async def start(app: web.Application) -> None:
        app["loop_task"] = asyncio.create_task(engine.run())

    async def stop(app: web.Application) -> None:
        app["loop_task"].cancel()

    app.on_startup.append(start)
    app.on_cleanup.append(stop)
    app.router.add_get("/health", lambda r: web.json_response({"status": "ok"}))
    app.router.add_get("/internal/stats", lambda r: web.json_response(engine.stats()))
    app.router.add_get("/metrics", lambda r: web.Response(text=metrics_text(engine)))
    app.router.add_post("/v1/completions", completions)
    app.router.add_post("/internal/score", score)
    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    for f in fields(MockConfig):
        default = MockConfig.__dataclass_fields__[f.name].default
        kind = parse_bool if isinstance(default, bool) else type(default)
        parser.add_argument(f"--{f.name.replace('_', '-')}", type=kind, default=default)
    args = vars(parser.parse_args())
    port = args.pop("port")
    try:
        import uvloop

        uvloop.install()
    except ImportError:
        pass
    web.run_app(make_app(MockConfig(**args)), host="127.0.0.1", port=port, print=None,
                access_log=None)


if __name__ == "__main__":
    main()
