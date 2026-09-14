import asyncio
import time

import pytest

from llm_serving_engine.config import EngineConfig, KVCacheConfig, ModelConfig, ServerConfig
from llm_serving_engine.engine import EngineUnavailable, InferenceEngine, sequence_from_ingress
from llm_serving_engine.model.decode_graph import decode_graph_buckets
from llm_serving_engine.model.sampling import SamplingParams
from llm_serving_engine.observability.metrics_export import REGISTRY
from llm_serving_engine.scheduling.allocator import BlockAllocator, ContiguousAllocator
from llm_serving_engine.scheduling.dispatch import ABORTED, DONE, output_channels
from llm_serving_engine.scheduling.scheduler import (
    ContinuousBatchedScheduler,
    StaticBatchedScheduler,
)

TOKEN = 7


class FakeTokenizer:
    def encode_prompt(self, text: str) -> list[int]:
        return [ord(c) for c in text]


class FakeModelRunner:
    """No weights: every sequence that owes a token gets TOKEN, finishing at max_tokens."""

    device = "cpu"

    def __init__(self):
        self.kv_pool_calls: list[tuple[int, int]] = []
        self.captured_buckets: list[list[int]] = []
        self.freed: list[int] = []
        self.fail_next_forward = False

    def forward(self, plan, seqs):
        if self.fail_next_forward:
            self.fail_next_forward = False
            raise RuntimeError("injected forward failure")
        results = []
        for entry in plan:
            seq = seqs[entry.seq_id]
            if entry.is_prefill_chunk and seq.status == "PREFILLING":
                continue
            finished = len(seq.generated_tokens) + 1 >= seq.sampling_params.max_tokens
            results.append((seq.seq_id, TOKEN, finished))
        return results

    def allocate_kv_pool(self, num_blocks, block_size):
        self.kv_pool_calls.append((num_blocks, block_size))

    def capture_decode_graphs(self, bucket_sizes):
        self.captured_buckets.append(bucket_sizes)

    def free(self, seq_id):
        self.freed.append(seq_id)


class FailingScheduler(ContinuousBatchedScheduler):
    def scheduler_step(self, running, waiting, allocator):
        if waiting:
            raise RuntimeError("injected scheduler failure")
        return super().scheduler_step(running, waiting, allocator)


def make_config(**overrides) -> EngineConfig:
    model = overrides.pop("model", ModelConfig())
    kv_cache = overrides.pop("kv_cache", KVCacheConfig())
    server = overrides.pop("server", ServerConfig())
    return EngineConfig(model=model, kv_cache=kv_cache, server=server, **overrides)


def make_engine(config: EngineConfig | None = None, **kwargs) -> InferenceEngine:
    kwargs.setdefault("model_runner", FakeModelRunner())
    return InferenceEngine(config=config or make_config(), tokenizer=FakeTokenizer(), **kwargs)


async def read_stream(q: asyncio.Queue) -> list:
    items = []
    while not items or items[-1] not in (DONE, ABORTED):
        items.append(await asyncio.wait_for(q.get(), timeout=5))
    return items


def wait_until(condition, timeout: float = 5) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "condition never held"
        time.sleep(0.005)


def test_submit_tokenizes_and_hands_off_to_the_scheduler_thread():
    engine = make_engine()
    submission = engine.submit("hi", SamplingParams())

    assert output_channels[submission.seq_id] is submission.output_queue
    assert submission.prompt_len == 2
    req = engine.ingress.get_nowait()
    assert req.seq_id == submission.seq_id
    assert req.prompt_tokens == [ord("h"), ord("i")]


def test_submit_bounds_the_output_queue_from_config():
    engine = make_engine(make_config(server=ServerConfig(output_queue_maxsize=2)))
    assert engine.submit("hi", SamplingParams()).output_queue.maxsize == 2


def test_seq_ids_are_never_reused_across_engines():
    first = make_engine().submit("a", SamplingParams()).seq_id
    second = make_engine().submit("a", SamplingParams()).seq_id
    assert first != second


def test_a_closed_engine_rejects_submissions():
    engine = make_engine()
    engine.close_ingress()
    with pytest.raises(EngineUnavailable):
        engine.submit("hi", SamplingParams())


def test_stop_joins_both_threads():
    engine = make_engine()
    engine.start()
    assert engine.healthy
    engine.stop()
    assert not any(t.is_alive() for t in engine._threads)


@pytest.mark.asyncio
async def test_a_request_streams_its_tokens_then_done_and_records_its_latencies():
    engine = make_engine()
    engine.bind_loop(asyncio.get_running_loop())
    before = REGISTRY.get_sample_value("llm_total_latency_seconds_count") or 0
    engine.start()
    try:
        submission = engine.submit("hi", SamplingParams(max_tokens=3))
        assert await read_stream(submission.output_queue) == [TOKEN, TOKEN, TOKEN, DONE]
        await asyncio.to_thread(wait_until, lambda: engine.is_idle)
    finally:
        engine.stop()
    assert REGISTRY.get_sample_value("llm_total_latency_seconds_count") == before + 1
    assert submission.seq_id in engine.model_runner.freed


@pytest.mark.asyncio
async def test_a_failed_forward_aborts_its_batch_and_the_engine_keeps_serving():
    runner = FakeModelRunner()
    runner.fail_next_forward = True
    engine = make_engine(model_runner=runner)
    engine.bind_loop(asyncio.get_running_loop())
    engine.start()
    try:
        failed = engine.submit("hi", SamplingParams(max_tokens=2))
        assert await read_stream(failed.output_queue) == [ABORTED]

        served = engine.submit("hi", SamplingParams(max_tokens=2))
        assert await read_stream(served.output_queue) == [TOKEN, TOKEN, DONE]
        await asyncio.to_thread(wait_until, lambda: engine.is_idle)
        assert engine.healthy
        assert engine.kv_utilization == 0.0
    finally:
        engine.stop()


@pytest.mark.asyncio
async def test_a_failed_scheduler_aborts_every_request_and_reports_unhealthy():
    engine = make_engine(scheduler=FailingScheduler())
    engine.bind_loop(asyncio.get_running_loop())
    engine.start()
    try:
        submission = engine.submit("hi", SamplingParams())
        assert await read_stream(submission.output_queue) == [ABORTED]
        assert not engine.healthy
        with pytest.raises(EngineUnavailable):
            engine.submit("hi", SamplingParams())
    finally:
        engine.stop()


@pytest.mark.asyncio
async def test_preempted_requests_still_stream_every_token():
    runner = FakeModelRunner()
    # Two 3-token prompts fill a block each; both cross into a new block on the same step,
    # and the one free block goes to the older request.
    engine = make_engine(allocator=BlockAllocator(num_blocks=3, block_size=4), model_runner=runner)
    engine.bind_loop(asyncio.get_running_loop())
    before = REGISTRY.get_sample_value("llm_preemptions_total") or 0
    engine.start()
    try:
        submissions = [engine.submit("abc", SamplingParams(max_tokens=6)) for _ in range(2)]
        streams = await asyncio.gather(*(read_stream(s.output_queue) for s in submissions))
    finally:
        engine.stop()

    assert streams == [[TOKEN] * 6 + [DONE]] * 2
    assert REGISTRY.get_sample_value("llm_preemptions_total") > before
    latest = submissions[1].seq_id
    assert runner.freed.count(latest) >= 2  # once per preemption, once on finish


@pytest.mark.asyncio
async def test_a_request_the_kv_pool_can_never_hold_is_aborted():
    engine = make_engine(allocator=BlockAllocator(num_blocks=1, block_size=4))
    engine.bind_loop(asyncio.get_running_loop())
    engine.start()
    try:
        submission = engine.submit("too long", SamplingParams())
        assert await read_stream(submission.output_queue) == [ABORTED]
        await asyncio.to_thread(wait_until, lambda: engine.is_idle)
    finally:
        engine.stop()


def test_scheduler_config_selects_the_static_scheduler():
    engine = make_engine(make_config(scheduler="static", static_batch_size=3))
    assert isinstance(engine.scheduler, StaticBatchedScheduler)
    assert engine.scheduler.max_running == 3


def test_scheduler_config_defaults_to_continuous_with_its_concurrency_cap():
    engine = make_engine(make_config(max_concurrent_sequences=8))
    assert isinstance(engine.scheduler, ContinuousBatchedScheduler)
    assert engine.scheduler.max_running == 8


def test_contiguous_allocator_matches_the_paged_pool_capacity():
    contiguous = make_engine(make_config(kv_allocator="contiguous"))
    paged = make_engine()
    assert isinstance(contiguous.allocator, ContiguousAllocator)
    assert isinstance(paged.allocator, BlockAllocator)
    assert contiguous.allocator.capacity_tokens == paged.allocator.num_blocks * 16


def test_the_kv_pool_matches_the_allocator():
    runner = FakeModelRunner()
    make_engine(allocator=BlockAllocator(num_blocks=10, block_size=4), model_runner=runner)
    assert runner.kv_pool_calls == [(10, 4)]


def test_contiguous_allocator_allocates_no_pool_and_captures_no_graphs():
    runner = FakeModelRunner()
    make_engine(make_config(kv_allocator="contiguous"), model_runner=runner)
    assert runner.kv_pool_calls == []
    assert runner.captured_buckets == []


def test_without_custom_kernels_there_is_no_pool():
    runner = FakeModelRunner()
    make_engine(make_config(model=ModelConfig(use_custom_kernels=False)), model_runner=runner)
    assert runner.kv_pool_calls == []


def test_decode_graphs_cover_the_largest_batch_the_scheduler_can_run():
    runner = FakeModelRunner()
    make_engine(make_config(max_concurrent_sequences=64), model_runner=runner)
    assert runner.kv_pool_calls  # graphs capture into the pool, so the pool comes first
    [buckets] = runner.captured_buckets
    assert max(buckets) >= 64


def test_use_cuda_graphs_off_skips_capture():
    runner = FakeModelRunner()
    make_engine(make_config(model=ModelConfig(use_cuda_graphs=False)), model_runner=runner)
    assert runner.captured_buckets == []


def test_decode_graph_buckets_pad_any_batch_by_under_2x():
    buckets = decode_graph_buckets(64)
    assert buckets == [1, 2, 4, 8, 16, 32, 64]
    assert decode_graph_buckets(5)[-1] == 8


def test_explicit_scheduler_and_allocator_override_the_config():
    scheduler = ContinuousBatchedScheduler()
    allocator = ContiguousAllocator(capacity_tokens=100)
    engine = make_engine(make_config(scheduler="static"), scheduler=scheduler, allocator=allocator)
    assert engine.scheduler is scheduler
    assert engine.allocator is allocator


def test_sequence_from_ingress_leaves_admit_time_to_the_scheduler():
    engine = make_engine()
    submission = engine.submit("hi", SamplingParams())
    req = engine.ingress.get_nowait()

    seq = sequence_from_ingress(req)

    assert seq.seq_id == submission.seq_id
    assert seq.status == "WAITING"
    assert seq.metrics.enqueue_time == req.arrival_time
    assert seq.metrics.admit_time is None  # schedule_latency must include the wait
