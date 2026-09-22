import asyncio
import threading

import pytest

from llm_serving_engine.config import EngineConfig, ModelConfig, ServerConfig
from llm_serving_engine.engine import (
    EngineUnavailable,
    InferenceEngine,
    InvalidPrompt,
    sequence_from_ingress,
)
from llm_serving_engine.model.decode_graph import decode_graph_buckets
from llm_serving_engine.model.sampling import SamplingParams
from llm_serving_engine.observability.metrics_export import REGISTRY
from llm_serving_engine.scheduling.allocator import BlockAllocator, ContiguousAllocator
from llm_serving_engine.scheduling.dispatch import ABORTED, DONE, output_channels
from llm_serving_engine.scheduling.scheduler import (
    ContinuousBatchedScheduler,
    StaticBatchedScheduler,
)
from tests.factories import (
    TOKEN,
    FakeModelRunner,
    FakeTokenizer,
    GatedModelRunner,
    make_config,
    read_stream,
    seq_ids,
    wait_until,
)


class FailingScheduler(ContinuousBatchedScheduler):
    def scheduler_step(self, running, waiting, allocator):
        if waiting:
            raise RuntimeError("injected scheduler failure")
        return super().scheduler_step(running, waiting, allocator)


def make_engine(config: EngineConfig | None = None, **kwargs) -> InferenceEngine:
    kwargs.setdefault("model_runner", FakeModelRunner())
    return InferenceEngine(config=config or make_config(), tokenizer=FakeTokenizer(), **kwargs)


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


@pytest.mark.parametrize("prompt", ["", "\u0100"])  # no tokens; token id 256 == vocab_size
def test_a_prompt_the_model_cannot_run_is_rejected_at_submit(prompt):
    engine = make_engine()
    with pytest.raises(InvalidPrompt):
        engine.submit(prompt, SamplingParams())
    assert engine.ingress.empty()


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
async def test_a_forward_failure_that_loses_the_device_fails_the_engine():
    runner = FakeModelRunner()
    runner.fail_next_forward = True
    runner.device_lost = True
    engine = make_engine(model_runner=runner)
    engine.bind_loop(asyncio.get_running_loop())
    engine.start()
    try:
        submission = engine.submit("hi", SamplingParams())
        assert await read_stream(submission.output_queue) == [ABORTED]
        await asyncio.to_thread(wait_until, lambda: not engine.healthy)
        with pytest.raises(EngineUnavailable):
            engine.submit("hi", SamplingParams())
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


@pytest.mark.asyncio
async def test_cancel_ends_a_running_sequence_and_frees_its_blocks():
    gate = threading.Event()
    runner = GatedModelRunner(gate)
    engine = make_engine(model_runner=runner)
    engine.bind_loop(asyncio.get_running_loop())
    engine.start()
    try:
        submission = engine.submit("hi", SamplingParams(max_tokens=256))
        await asyncio.to_thread(wait_until, lambda: submission.seq_id in seq_ids(engine.running))
        engine.cancel(submission.seq_id)
        gate.set()
        # The iteration already in flight when cancel arrived may still deliver its
        # token; cancellation only stops the sequence from being planned again.
        stream = await read_stream(submission.output_queue)
        assert stream[-1] == ABORTED
        assert len(stream) < 256
        await asyncio.to_thread(wait_until, lambda: engine.is_idle)
    finally:
        gate.set()
        engine.stop()
    assert engine.kv_utilization == 0.0
    assert submission.seq_id in engine.model_runner.freed


@pytest.mark.asyncio
async def test_cancel_drops_a_waiting_sequence_before_it_ever_runs():
    engine = make_engine(make_config(scheduler="static", static_batch_size=1))
    engine.bind_loop(asyncio.get_running_loop())
    engine.start()
    try:
        # A static batch of 1 admits blocking alone; queued stays behind it in `waiting`
        # until it finishes. Its huge max_tokens never binds first: the KV pool exhausts
        # and rejects it (`can_ever_fit` on a sequence this long) long before that.
        blocking = engine.submit("hi", SamplingParams(max_tokens=10**9))
        await asyncio.to_thread(wait_until, lambda: blocking.seq_id in seq_ids(engine.running))
        queued = engine.submit("hi", SamplingParams(max_tokens=256))
        await asyncio.to_thread(wait_until, lambda: queued.seq_id in seq_ids(engine.waiting))

        engine.cancel(queued.seq_id)

        assert await read_stream(queued.output_queue) == [ABORTED]
        assert queued.seq_id not in seq_ids(engine.running)
        assert queued.seq_id not in seq_ids(engine.waiting)
        assert engine.healthy
    finally:
        engine.stop()


@pytest.mark.asyncio
async def test_cancel_of_an_already_finished_sequence_is_a_no_op():
    engine = make_engine()
    engine.bind_loop(asyncio.get_running_loop())
    engine.start()
    try:
        submission = engine.submit("hi", SamplingParams(max_tokens=1))
        assert await read_stream(submission.output_queue) == [TOKEN, DONE]
        await asyncio.to_thread(wait_until, lambda: engine.is_idle)

        engine.cancel(submission.seq_id)
        await asyncio.to_thread(wait_until, lambda: engine.ingress.empty())

        assert engine.is_idle
        assert engine.healthy
        assert submission.output_queue.empty()  # no second ABORTED behind the DONE already read
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


def test_the_kv_pool_matches_the_allocator_and_exists_before_graph_capture():
    runner = FakeModelRunner()
    make_engine(allocator=BlockAllocator(num_blocks=10, block_size=4), model_runner=runner)
    captures = ["capture_decode_graphs", "capture_piecewise_graphs"]
    assert [call[0] for call in runner.setup_calls] == ["allocate_kv_pool", *captures]
    assert runner.setup_calls[0] == ("allocate_kv_pool", 10, 4)


def test_contiguous_allocator_allocates_no_pool_and_captures_no_graphs():
    runner = FakeModelRunner()
    make_engine(make_config(kv_allocator="contiguous"), model_runner=runner)
    assert runner.setup_calls == []


def test_without_custom_kernels_there_is_no_pool():
    runner = FakeModelRunner()
    make_engine(make_config(model=ModelConfig(use_custom_kernels=False)), model_runner=runner)
    assert runner.setup_calls == []


def test_graphs_cover_the_largest_batch_and_token_count_the_scheduler_can_plan():
    runner = FakeModelRunner()
    make_engine(make_config(max_concurrent_sequences=64, token_budget=4096), model_runner=runner)
    buckets = dict(runner.setup_calls[1:])
    assert max(buckets["capture_decode_graphs"]) >= 64
    assert max(buckets["capture_piecewise_graphs"]) >= 4096 + 64


def test_use_cuda_graphs_off_skips_capture():
    runner = FakeModelRunner()
    make_engine(make_config(model=ModelConfig(use_cuda_graphs=False)), model_runner=runner)
    assert [call[0] for call in runner.setup_calls] == ["allocate_kv_pool"]


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
