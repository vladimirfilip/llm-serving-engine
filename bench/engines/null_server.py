"""A server that streams tokens on a fixed cadence with pre-encoded events, so its own
timing is negligible. It exists to measure the client: `python -m bench.engines.null_server
--port N --gap-ms 10`. One global tick writes one token to every open stream."""

from __future__ import annotations

import argparse
import asyncio
import json
import multiprocessing
import time

TEXT_EVENT = b'data: {"choices":[{"index":0,"text":" t","finish_reason":null}]}\n\n'


def chunked(payload: bytes) -> bytes:
    return b"%x\r\n" % len(payload) + payload + b"\r\n"


class Stream:
    __slots__ = ("left", "max_tokens", "prompt_tokens", "writer")

    def __init__(self, writer: asyncio.StreamWriter, max_tokens: int, prompt_tokens: int):
        self.writer, self.left = writer, max_tokens
        self.max_tokens, self.prompt_tokens = max_tokens, prompt_tokens


async def serve(port: int, gap_s: float, reuse_port: bool) -> None:
    streams: list[Stream] = []
    event = chunked(TEXT_EVENT)

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        while True:
            try:
                head = await reader.readuntil(b"\r\n\r\n")
            except (asyncio.IncompleteReadError, ConnectionError):
                return
            length = next(int(h.split(b":")[1]) for h in head.split(b"\r\n")
                          if h.lower().startswith(b"content-length"))
            body = json.loads(await reader.readexactly(length))
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                         b"Transfer-Encoding: chunked\r\n\r\n")
            prompt = body["prompt"]
            streams.append(Stream(writer, body["max_tokens"], len(prompt)))

    async def tick() -> None:
        deadline = time.perf_counter()
        while True:
            deadline += gap_s
            await asyncio.sleep(max(0.0, deadline - time.perf_counter() - 0.001))
            while time.perf_counter() < deadline:
                await asyncio.sleep(0)
            finished = []
            for stream in streams:
                stream.writer.write(event)
                stream.left -= 1
                if stream.left == 0:
                    usage = json.dumps({"choices": [], "usage": {
                        "prompt_tokens": stream.prompt_tokens,
                        "completion_tokens": stream.max_tokens}}).encode()
                    stream.writer.write(chunked(b"data: " + usage + b"\n\n")
                                        + chunked(b"data: [DONE]\n\n") + b"0\r\n\r\n")
                    finished.append(stream)
            for stream in finished:
                streams.remove(stream)

    server = await asyncio.start_server(handle, "127.0.0.1", port, reuse_port=reuse_port,
                                        backlog=1024)
    async with server:
        await asyncio.gather(server.serve_forever(), tick())


def worker(port: int, gap_ms: float, reuse_port: bool) -> None:
    try:
        import uvloop

        uvloop.install()
    except ImportError:
        pass
    asyncio.run(serve(port, gap_ms / 1000, reuse_port))


def main() -> None:
    """`--workers N` processes share the port, so the server's own write rate is not the limit
    on what the client can be shown."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--gap-ms", type=float, default=10.0)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    children = [multiprocessing.Process(target=worker, args=(args.port, args.gap_ms, True),
                                        daemon=True) for _ in range(args.workers - 1)]
    for child in children:
        child.start()
    worker(args.port, args.gap_ms, args.workers > 1)


if __name__ == "__main__":
    main()
