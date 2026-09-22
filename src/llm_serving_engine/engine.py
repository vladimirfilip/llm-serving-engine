"""Wires the IO thread to the scheduler and GPU-worker threads.

The IO thread (asyncio) owns `submit` and `cancel`. The scheduler thread drains `ingress`
(requests and cancellations both, in the order they arrived) and applies each iteration's
results. The GPU worker thread runs `model_runner.forward` on each plan. Every hop is a
`queue.SimpleQueue`: a blocked `get` releases the GIL and returns as soon as the other side
puts, so no thread polls, and `stop` puts None on `ingress` and `_plan_queue` to wake
whichever thread is waiting on each.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import torch

from .config import EngineConfig
from .model.decode_graph import decode_graph_buckets
from .model.model_runner import IterationResults, ModelRunner
from .model.piecewise_graph import piecewise_graph_buckets
from .model.sampling import SamplingParams
from .model.tokenizer import TokenizerWrapper
from .observability.metrics import RequestMetrics
from .observability.metrics_export import PREEMPTIONS_TOTAL, REQUESTS_IN_FLIGHT, record_request
from .observability.nvtx import nvtx_range
from .scheduling.allocator import BlockAllocator, ContiguousAllocator, KVAllocator
from .scheduling.batch_plan import BatchPlan
from .scheduling.dispatch import new_output_channel
from .scheduling.scheduler import ContinuousBatchedScheduler, Scheduler, StaticBatchedScheduler
from .scheduling.sequence import Sequence

logger = logging.getLogger(__name__)


class EngineUnavailable(RuntimeError):
    """The engine accepts no new requests: it is stopping, switching models, or failed."""


class InvalidPrompt(ValueError):
    """A prompt the model can't run: no tokens, or token ids outside its vocabulary."""


class DeviceLost(RuntimeError):
    """A CUDA error left this process's device context unusable; only a restart recovers."""


@dataclass(slots=True)
class IngressRequest:
    """Plain values crossing from the IO thread to the scheduler thread; the scheduler
    thread builds the Sequence."""

    seq_id: int
    prompt_tokens: list[int]
    sampling_params: SamplingParams
    arrival_time: float
    logprobs: list[float] | None = None


@dataclass(slots=True)
class CancelRequest:
    """Crosses `ingress` behind whatever `IngressRequest` admits the same seq_id, so a
    cancellation can never overtake -- and silently miss -- the admission it targets."""

    seq_id: int


@dataclass(slots=True)
class Submission:
    seq_id: int
    output_queue: asyncio.Queue
    prompt_len: int
    tokenizer: TokenizerWrapper  # the tokenizer that encoded the prompt decodes its tokens
    logprobs: list[float] | None = None  # grows one entry per token before that token is queued
    # Ends this request if its stream stops reading before DONE or ABORTED.
    cancel: Callable[[], None] = lambda: None


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
        # Decode tokens don't draw on the token budget, so an iteration can carry both.
        max_tokens = self.scheduler.token_budget + self.scheduler.max_running
        graphs = config.model.use_cuda_graphs
        decode_buckets = decode_graph_buckets(self.scheduler.max_running) if graphs else []
        piecewise_buckets = piecewise_graph_buckets(max_tokens) if graphs else []
        paged = config.kv_allocator == "paged"
        if allocator is None:
            reserved = 0
            if config.model.use_custom_kernels and model_runner.device.startswith("cuda"):
                # Graphs replay only against a paged pool, but every fused iteration needs its
                # activation peak free, whichever allocator holds the KV.
                reserved = model_runner.bytes_beyond_kv_pool(
                    config.kv_cache.block_size,
                    max_tokens,
                    decode_buckets if paged else [],
                    piecewise_buckets if paged else [],
                )
            allocator = _build_allocator(config, model_runner, reserved)
        self.allocator = allocator
        if isinstance(self.allocator, BlockAllocator) and config.model.use_custom_kernels:
            model_runner.allocate_kv_pool(self.allocator.num_blocks, self.allocator.block_size)
            if graphs:
                model_runner.capture_decode_graphs(decode_buckets)
                model_runner.capture_piecewise_graphs(piecewise_buckets)
        self.ingress: queue.SimpleQueue[IngressRequest | CancelRequest | None] = queue.SimpleQueue()
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []
        self._plan_queue: queue.SimpleQueue[tuple[BatchPlan, dict[int, Sequence]] | None] = (
            queue.SimpleQueue()
        )
        self._results_queue: queue.SimpleQueue[IterationResults | Exception] = queue.SimpleQueue()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        # Taken around the accepting check and the ingress put, so closing the ingress can't
        # race a submit into an engine that no longer drains it.
        self._ingress_lock = threading.Lock()
        self._accepting = True
        self._failed = False
        # Each written by one thread only; the engine is idle when they are equal.
        self._submitted = 0
        self._ended = 0
        self._preemptions = 0

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Called once from the IO thread at startup; every token callback targets `loop`."""
        self._loop = loop

    def submit(self, prompt: str, sampling_params: SamplingParams) -> Submission:
        """IO thread: tokenize, open the output channel, hand off to the scheduler thread.
        The caller streams from `output_queue` until DONE or ABORTED, and pops its
        `output_channels` entry when it stops reading."""
        return self.submit_tokens(self.tokenizer.encode_prompt(prompt), sampling_params)

    def submit_tokens(
        self, prompt_tokens: list[int], sampling_params: SamplingParams, logprobs: bool = False
    ) -> Submission:
        """`submit` for a caller that has already tokenized, chat template and BOS included."""
        # An out-of-vocabulary id faults the embedding lookup on the device, which takes the
        # whole engine down with it.
        if not prompt_tokens or max(prompt_tokens) >= self.model_runner.vocab_size:
            raise InvalidPrompt(
                f"prompt encodes to no tokens or to ids outside the model's "
                f"{self.model_runner.vocab_size}-token vocabulary"
            )
        token_logprobs = [] if logprobs else None
        with self._ingress_lock:
            if not self._accepting:
                raise EngineUnavailable("engine is not accepting requests")
            seq_id, output_queue = new_output_channel(self.config.server.output_queue_maxsize)
            self._submitted += 1
            self.ingress.put(
                IngressRequest(
                    seq_id, prompt_tokens, sampling_params, time.monotonic(), token_logprobs
                )
            )
        return Submission(
            seq_id, output_queue, len(prompt_tokens), self.tokenizer, token_logprobs,
            cancel=partial(self.cancel, seq_id),
        )

    def cancel(self, seq_id: int) -> None:
        """Any thread, even after `close_ingress`: ends `seq_id`'s stream and frees its
        blocks once the scheduler thread next drains `ingress`. Queued behind whatever
        `IngressRequest` admitted `seq_id`, so it never overtakes -- and never misses -- a
        request still waiting to be admitted. A no-op once `seq_id` has already ended."""
        self.ingress.put(CancelRequest(seq_id))

    def close_ingress(self) -> None:
        with self._ingress_lock:
            self._accepting = False

    @property
    def is_idle(self) -> bool:
        """Every submitted request has ended."""
        return self._ended == self._submitted

    def stats(self) -> dict:
        """Cheap snapshot for a 20 Hz poller: scheduler occupancy, KV blocks and where device
        memory went. Read from another thread, so counts may lag a step."""
        running, waiting = list(self.running), len(self.waiting)
        allocator = self.allocator
        block_size = self.config.kv_cache.block_size
        blocks_total = getattr(allocator, "num_blocks", None) or (
            allocator.capacity_tokens // block_size
        )
        blocks_used = round(allocator.utilization * blocks_total)
        return {
            "running": len(running),
            "waiting": waiting,
            "preemptions_total": self._preemptions,
            "kv": {
                "block_size": block_size,
                "blocks_total": blocks_total,
                "blocks_used": blocks_used,
                "tokens_used": sum(seq.block_table.num_tokens for seq in running),
            },
            "memory_bytes": self.model_runner.memory_bytes(),
        }

    @property
    def healthy(self) -> bool:
        return not self._failed and all(t.is_alive() for t in self._threads)

    @property
    def kv_utilization(self) -> float:
        return self.allocator.utilization

    def start(self) -> None:
        scheduler = threading.Thread(target=self._scheduler_loop, name="scheduler", daemon=True)
        worker = threading.Thread(target=self._gpu_worker_loop, name="gpu-worker", daemon=True)
        self._threads = [scheduler, worker]
        scheduler.start()
        worker.start()

    def stop(self) -> None:
        self.close_ingress()
        self._stop.set()
        self.ingress.put(None)
        self._plan_queue.put(None)
        for t in self._threads:
            t.join(timeout=5)

    def _scheduler_loop(self) -> None:
        try:
            self._schedule_until_stopped()
        except Exception:
            logger.exception("engine failed; aborting every request")
            self._failed = True
            self.close_ingress()
            self._drain_ingress()
            self._abort(self.running + list(self.waiting))
            self.waiting.clear()

    def _schedule_until_stopped(self) -> None:
        """Ping-pongs one BatchPlan at a time: a decode entry's input is the token the
        previous iteration produced, which exists only once its results are applied."""
        in_flight: tuple[BatchPlan, dict[int, Sequence]] | None = None
        while not self._stop.is_set():
            with nvtx_range("step"):
                if in_flight is not None:
                    outcome = self._results_queue.get()
                    with nvtx_range("postprocess"):
                        self._apply(outcome, *in_flight)
                    in_flight = None
                self._drain_ingress()

                with nvtx_range("schedule"):
                    plan = self.scheduler.scheduler_step(self.running, self.waiting, self.allocator)
                    for seq_id in plan.preempted:
                        self.model_runner.free(seq_id)
                    PREEMPTIONS_TOTAL.inc(len(plan.preempted))
                    self._preemptions += len(plan.preempted)
                    if plan.rejected:
                        self._abort(plan.rejected)
                    REQUESTS_IN_FLIGHT.set(len(self.running))

                if len(plan):
                    in_flight = (plan, {seq.seq_id: seq for seq in self.running})
                    self._plan_queue.put(in_flight)
                else:
                    # Nothing runnable until a request arrives: the scheduler plans something
                    # whenever `running` is non-empty, so an empty plan leaves nothing running.
                    self._process_ingress(self.ingress.get())

    def _apply(
        self, outcome: IterationResults | Exception, plan: BatchPlan, seqs: dict[int, Sequence]
    ) -> None:
        """Folds one iteration back into scheduler state. A failed batch had already
        advanced its sequences' prefill progress and block tables, so every sequence in it
        is aborted: retrying would run against state the model never computed."""
        if isinstance(outcome, DeviceLost):
            raise outcome
        if isinstance(outcome, Exception):
            self._abort([seqs[entry.seq_id] for entry in plan])
            return
        finished = self.scheduler.handle_iteration_results(
            outcome, self.running, self.allocator, self._loop
        )
        for seq in finished:
            self._end(seq)

    def _abort(self, seqs: list[Sequence]) -> None:
        self.scheduler.abort(seqs, self.running, self.allocator, self._loop)
        for seq in seqs:
            self._end(seq)

    def _end(self, seq: Sequence) -> None:
        self.model_runner.free(seq.seq_id)
        record_request(seq.metrics)
        self._ended += 1

    def _drain_ingress(self) -> None:
        while True:
            try:
                self._process_ingress(self.ingress.get_nowait())
            except queue.Empty:
                return

    def _process_ingress(self, item: IngressRequest | CancelRequest | None) -> None:
        if isinstance(item, IngressRequest):
            self.waiting.append(sequence_from_ingress(item))
        elif isinstance(item, CancelRequest):
            self._cancel(item.seq_id)

    def _cancel(self, seq_id: int) -> None:
        """A no-op once `seq_id` has already ended, since `cancel` may reach here after
        the scheduler thread has already applied its DONE or aborted it some other way."""
        seq = self._find_unfinished(seq_id)
        if seq is None:
            return
        if seq in self.waiting:
            self.waiting.remove(seq)
        self._abort([seq])

    def _find_unfinished(self, seq_id: int) -> Sequence | None:
        for seq in self.running:
            if seq.seq_id == seq_id:
                return seq
        for seq in self.waiting:
            if seq.seq_id == seq_id:
                return seq
        return None

    def _gpu_worker_loop(self) -> None:
        while (item := self._plan_queue.get()) is not None:
            plan, seqs = item
            try:
                outcome: IterationResults | Exception = self.model_runner.forward(plan, seqs)
            except Exception as e:
                logger.exception("forward pass failed")
                outcome = e if self.model_runner.device_usable() else DeviceLost(str(e))
            self._results_queue.put(outcome)


def _build_scheduler(config: EngineConfig) -> Scheduler:
    if config.scheduler == "static":
        return StaticBatchedScheduler(
            batch_size=config.static_batch_size, token_budget=config.token_budget
        )
    return ContinuousBatchedScheduler(
        token_budget=config.token_budget,
        max_concurrent_sequences=config.max_concurrent_sequences,
    )


def _build_allocator(
    config: EngineConfig, model_runner: ModelRunner, reserved_bytes: int
) -> KVAllocator:
    """Sizes the pool from free memory less `reserved_bytes`: the largest eager iteration's
    activations, plus a paged run's CUDA graphs. A contiguous run allocates each sequence's
    buffer from the resulting capacity at admission."""
    block_size = config.kv_cache.block_size
    free = _capped_free_bytes(model_runner, config.kv_cache.device_memory_fraction)
    num_blocks = config.kv_cache.num_blocks(free - reserved_bytes)
    if config.kv_allocator == "contiguous":
        return ContiguousAllocator(capacity_tokens=num_blocks * block_size)
    return BlockAllocator(num_blocks=num_blocks, block_size=block_size)


def _free_memory_bytes(model_runner: ModelRunner) -> int:
    """Free VRAM on GPU; a fixed 2 GiB budget on CPU, which has no equivalent to query."""
    if model_runner.device.startswith("cuda"):
        return torch.cuda.mem_get_info()[0]
    return 2 * 1024**3


def _capped_free_bytes(model_runner: ModelRunner, device_fraction: float | None) -> int:
    """Free VRAM, held to what keeps everything on the device within `device_fraction` of its
    total: the cap leaves `total * fraction - already_used` for what this engine adds."""
    free = _free_memory_bytes(model_runner)
    if device_fraction is None or not model_runner.device.startswith("cuda"):
        return free
    total = torch.cuda.mem_get_info()[1]
    return min(free, int(total * device_fraction) - (total - free))


def sequence_from_ingress(req: IngressRequest) -> Sequence:
    """A WAITING Sequence. admit_time stays unset until the scheduler admits it, so
    schedule_latency covers the time spent in `waiting`."""
    return Sequence(
        seq_id=req.seq_id,
        prompt_tokens=req.prompt_tokens,
        sampling_params=req.sampling_params,
        metrics=RequestMetrics(enqueue_time=req.arrival_time),
        logprobs=req.logprobs,
    )
