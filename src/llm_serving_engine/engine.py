"""Wires the IO thread to the scheduler/GPU-worker threads.

Three threads: the IO thread (asyncio, elsewhere) owns `submit`/`cancel`; the scheduler
thread here drains `ingress`, runs `scheduler_step`, and applies `handle_iteration_results`;
the GPU worker thread runs `model_runner.forward` on each plan. The two hops between
them are plain `deque`s, not `queue.Queue`: append/popleft are GIL-atomic, so a single
producer and a single consumer need no lock around them.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass

from .allocator import BlockAllocator, ContiguousAllocator
from .batch_plan import BatchPlan
from .config import EngineConfig
from .dispatch import new_output_channel, output_channels
from .metrics import RequestMetrics
from .model_runner import ModelRunner
from .sampling import SamplingParams
from .scheduler import ContinuousBatchedScheduler, Scheduler, StaticBatchedScheduler
from .sequence import Sequence
from .tokenizer import TokenizerWrapper

KVAllocator = BlockAllocator | ContiguousAllocator

# An empty-deque spin with no blocking call starves the sibling thread under the GIL:
# a CPU-bound Python loop doesn't yield often enough for the GPU worker's CUDA launches
# to get scheduled. This sleep is what releases the GIL between polls.
_IDLE_POLL_S = 0.0005


@dataclass(slots=True)
class IngressRequest:
    """Plain values crossing the ingress queue from the IO thread to the scheduler
    thread — no live Sequence object, just what's needed to build one."""

    seq_id: int
    prompt_tokens: list[int]
    sampling_params: SamplingParams
    arrival_time: float


class InferenceEngine:
    def __init__(
        self,
        config: EngineConfig,
        tokenizer: TokenizerWrapper,
        model_runner: ModelRunner,
        scheduler: Scheduler | None = None,
        allocator: KVAllocator | None = None,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.model_runner = model_runner
        self.scheduler = scheduler if scheduler is not None else _build_scheduler(config)
        num_blocks = config.kv_cache.num_blocks(_free_memory_bytes(model_runner))
        self.allocator = allocator if allocator is not None else _build_allocator(config, num_blocks)
        if config.kv_allocator == "paged" and config.model.use_custom_kernels:
            model_runner.allocate_kv_pool(num_blocks, config.kv_cache.block_size)
        self.ingress: queue.SimpleQueue[IngressRequest] = queue.SimpleQueue()
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []
        self._plan_queue: deque[tuple[BatchPlan, dict[int, Sequence]]] = deque()
        self._results_queue: deque[list[tuple[int, int, bool]]] = deque()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._next_seq_id = 0
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Called once from the IO thread at startup; the GPU worker targets this loop
        for every call_soon_threadsafe."""
        self._loop = loop

    def submit(self, prompt: str, sampling_params: SamplingParams) -> tuple[int, asyncio.Queue]:
        """IO thread: tokenize, create the output channel, hand off to the ingress queue.

        Returns (seq_id, output_queue). The caller streams from output_queue until the
        DONE sentinel (llm_serving_engine.dispatch.DONE) and should pop its own entry
        in a finally block so a disconnected client's channel doesn't linger.
        """
        seq_id = self._next_seq_id
        self._next_seq_id += 1
        prompt_tokens = self.tokenizer.encode_prompt(prompt)
        q = new_output_channel(seq_id, self.config.server.output_queue_maxsize)
        self.ingress.put(
            IngressRequest(
                seq_id=seq_id,
                prompt_tokens=prompt_tokens,
                sampling_params=sampling_params,
                arrival_time=time.monotonic(),
            )
        )
        return seq_id, q

    def cancel(self, seq_id: int) -> None:
        """Client disconnected before the stream finished. Drops the output channel;
        the sequence itself is only known to the scheduler thread, so freeing its blocks
        still routes through the same result-handling path other finishes do."""
        output_channels.pop(seq_id, None)

    def start(self) -> None:
        """Spawns the scheduler and GPU worker threads."""
        scheduler = threading.Thread(target=self._scheduler_loop, name="scheduler", daemon=True)
        worker = threading.Thread(target=self._gpu_worker_loop, name="gpu-worker", daemon=True)
        self._threads = [scheduler, worker]
        scheduler.start()
        worker.start()

    def stop(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=5)

    def _drain_ingress(self) -> None:
        while True:
            try:
                req = self.ingress.get_nowait()
            except queue.Empty:
                return
            self.waiting.append(sequence_from_ingress(req))

    def _scheduler_loop(self) -> None:
        """Ping-pongs one BatchPlan in flight at a time: a decode entry's input is
        `seq.generated_tokens[-1]`, which only reflects the prior iteration once
        handle_iteration_results has run, so the next plan can't be built until the
        previous one's results are back (no free pipelining across iterations here)."""
        in_flight = False
        while not self._stop.is_set():
            self._drain_ingress()
            if in_flight:
                try:
                    iter_results = self._results_queue.popleft()
                except IndexError:
                    time.sleep(_IDLE_POLL_S)
                    continue
                self.scheduler.handle_iteration_results(
                    iter_results, self.running, self.allocator, self._loop
                )
                for seq_id, _token, finished in iter_results:
                    if finished:
                        self.model_runner.free(seq_id)
                in_flight = False
            plan = self.scheduler.scheduler_step(self.running, self.waiting, self.allocator)
            if len(plan):
                seqs = {seq.seq_id: seq for seq in self.running}
                self._plan_queue.append((plan, seqs))
                in_flight = True
            elif not in_flight:
                time.sleep(_IDLE_POLL_S)

    def _gpu_worker_loop(self) -> None:
        while not self._stop.is_set():
            try:
                plan, seqs = self._plan_queue.popleft()
            except IndexError:
                time.sleep(_IDLE_POLL_S)
                continue
            try:
                results = self.model_runner.forward(plan, seqs)
            except Exception:
                # An uncaught exception here would otherwise kill this thread silently,
                # leaving the scheduler waiting forever for results that never arrive —
                # one bad batch would hang every client, not just the one that caused it.
                logging.getLogger(__name__).exception("GPU worker dropped a batch")
                results = []
            self._results_queue.append(results)


def _build_scheduler(config: EngineConfig) -> Scheduler:
    if config.scheduler == "static":
        return StaticBatchedScheduler(
            batch_size=config.static_batch_size, token_budget=config.token_budget
        )
    return ContinuousBatchedScheduler(
        token_budget=config.token_budget,
        max_concurrent_sequences=config.max_concurrent_sequences,
    )


def _build_allocator(config: EngineConfig, num_blocks: int) -> KVAllocator:
    """Both allocators are sized off the same `num_blocks`, so a run picking
    "contiguous" reserves the identical total token budget a "paged" run would —
    comparing the two at matched memory rather than matched block count."""
    if config.kv_allocator == "contiguous":
        return ContiguousAllocator(capacity_tokens=num_blocks * config.kv_cache.block_size)
    return BlockAllocator(num_blocks=num_blocks, block_size=config.kv_cache.block_size)


def _free_memory_bytes(model_runner: ModelRunner) -> int:
    """Sizing input for KVCacheConfig.num_blocks: real free VRAM on GPU, a fixed
    dev-mode budget on CPU where there's no equivalent signal to query."""
    if model_runner.device.startswith("cuda"):
        import torch

        return torch.cuda.mem_get_info()[0]
    return 2 * 1024**3


def sequence_from_ingress(req: IngressRequest) -> Sequence:
    """Turns a queued plain-value request into scheduler-visible Sequence state.

    admit_time stays None here: this sequence is only entering `waiting`, and
    schedule_latency must cover that wait, so admit_time is stamped later, when the
    scheduler actually admits the sequence.
    """
    metrics = RequestMetrics(enqueue_time=req.arrival_time)
    return Sequence(
        seq_id=req.seq_id,
        prompt_tokens=req.prompt_tokens,
        sampling_params=req.sampling_params,
        arrival_time=req.arrival_time,
        metrics=metrics,
    )
