"""One streaming completion, timed by the client. Times are `perf_counter` seconds."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from functools import lru_cache

import aiohttp
import numpy as np
import orjson

from ..engines.base import request_body
from ..workloads.workloads import Request

HEADERS = {"Content-Type": "application/json"}


@dataclass(frozen=True, slots=True)
class Target:
    """Everything the client needs to send a request to one engine; picklable so client
    processes can be handed one."""

    url: str  # base url
    model: str
    extra_body: dict
    accepts_token_ids: bool
    timeout_s: float
    tokenizer_path: str | None = None  # only for engines that take text
    logprobs: bool = False


@lru_cache(maxsize=1)
def _tokenizer(path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path)


def prompt_for(target: Target, prompt_ids: list[int]) -> list[int] | str:
    """Token ids when the engine takes them. Otherwise the text of everything after the BOS,
    which the server prepends again."""
    if target.accepts_token_ids:
        return prompt_ids
    return _tokenizer(target.tokenizer_path).decode(prompt_ids[1:])


def new_session() -> aiohttp.ClientSession:
    return aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=0), timeout=aiohttp.ClientTimeout(total=None)
    )


def blank_record(req: Request, t_sched: float) -> dict:
    return {"req_id": req.req_id, "t_sched": t_sched, "t_send": np.nan, "t_first": np.nan,
            "t_done": np.nan, "prompt_len": len(req.prompt_token_ids),
            "output_len_req": req.output_len,
            "prompt_tokens_usage": 0, "completion_tokens_usage": 0, "n_chunks": 0,
            "token_times": [], "token_logprobs": [], "token_ids": [], "status": "ok",
            "http_status": 0, "error": ""}


async def send(session: aiohttp.ClientSession, target: Target, req: Request, t_sched: float,
               t0: float, open_loop: bool = True) -> dict:
    """Sends `req` at `t0 + t_sched` (open loop) or at once (closed loop) and returns its
    record, times relative to `t0`. `t_send` is stamped immediately before the request goes out."""
    rec = blank_record(req, t_sched)
    body = request_body(target.model, prompt_for(target, req.prompt_token_ids), req.output_len,
                        target.extra_body, logprobs=target.logprobs)
    payload = orjson.dumps(body)
    if open_loop:
        await asyncio.sleep(max(0.0, t0 + t_sched - time.perf_counter()))
    times: list[float] = []
    rec["t_send"] = time.perf_counter() - t0
    if not open_loop:
        rec["t_sched"] = rec["t_send"]
    try:
        async with session.post(target.url + "/v1/completions", data=payload, headers=HEADERS,
                                timeout=aiohttp.ClientTimeout(total=target.timeout_s)) as resp:
            rec["http_status"] = resp.status
            if resp.status != 200:
                rec.update(status="error", error=(await resp.text())[:200])
                return rec
            await _read_stream(resp, rec, times, t0, target.logprobs)
    except asyncio.TimeoutError:
        rec.update(status="timeout", error="client timeout")
    except (aiohttp.ClientError, ConnectionError) as e:
        rec.update(status="error", error=f"{type(e).__name__}: {e}"[:200])
    rec["token_times"] = times
    rec["n_chunks"] = len(times)
    if rec["status"] == "ok" and not times:
        rec.update(status="error", error="no tokens streamed")
    if times:
        rec["t_first"] = times[0]
    return rec


async def _read_stream(
    resp: aiohttp.ClientResponse, rec: dict, times: list[float], t0: float, parse_tokens: bool
) -> None:
    """Reads whole network chunks and splits the SSE lines out of each. Every event in a chunk
    reached the client together, so they share the chunk's arrival time; this keeps the
    per-event cost low enough for hundreds of concurrent streams."""
    pending = b""
    async for chunk in resp.content.iter_any():
        now = time.perf_counter() - t0
        pending += chunk
        *lines, pending = pending.split(b"\n")
        for line in lines:
            if line.startswith(b"data:") and _on_event(line[5:].lstrip(), rec, times, now,
                                                       parse_tokens):
                return
    rec.update(status="error", error="stream ended without [DONE]")


EMPTY_TEXT = (b'"text":""', b'"text": ""')


def _on_event(payload: bytes, rec: dict, times: list[float], now: float,
              parse_tokens: bool) -> bool:
    """Records one SSE event; True once the stream is finished. A token event is only
    counted by a substring test unless its logprobs are wanted: parsing every token's JSON
    would cost more CPU than the timing it protects."""
    if payload.startswith(b"[DONE]"):
        rec["t_done"] = now
        return True
    if not parse_tokens and b'"usage"' not in payload:
        if b'"text"' in payload and not any(marker in payload for marker in EMPTY_TEXT):
            times.append(now)
        elif b'"error"' in payload:
            rec.update(status="error", error=payload[:200].decode(errors="replace"))
        return False
    event = orjson.loads(payload)
    if error := event.get("error"):
        rec.update(status="error", error=str(error)[:200])
        return False
    if usage := event.get("usage"):
        rec["prompt_tokens_usage"] = usage["prompt_tokens"]
        rec["completion_tokens_usage"] = usage["completion_tokens"]
    if choices := event.get("choices"):
        choice = choices[0]
        if choice.get("text"):
            times.append(now)
        if (lp := choice.get("logprobs")) and lp.get("token_logprobs"):
            rec["token_logprobs"].append(lp["token_logprobs"][0])
            token = lp.get("tokens", [""])[0]
            if token.startswith("token_id:"):  # engines that return plain text have no ids
                rec["token_ids"].append(int(token.removeprefix("token_id:")))
    return False
