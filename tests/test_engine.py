import asyncio
import time

import pytest

from llm_serving_engine.allocator import BlockAllocator, ContiguousAllocator
from llm_serving_engine.config import EngineConfig, KVCacheConfig, ModelConfig, ServerConfig
from llm_serving_engine.dispatch import output_channels
from llm_serving_engine.engine import InferenceEngine, sequence_from_ingress
from llm_serving_engine.sampling import SamplingParams
from llm_serving_engine.scheduler import ContinuousBatchedScheduler, StaticBatchedScheduler


class FakeTokenizer:
    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def encode_prompt(self, text: str) -> list[int]:
        return self.encode(text)


class FakeModelRunner:
    """Stands in for ModelRunner: no real weights, so forward() reports nothing rather
    than sampling, but the GPU worker loop still has a real callable to drive."""

    device = "cpu"

    def __init__(self):
        self.kv_pool_calls = []
        self.graph_capture_calls = 0

    def forward(self, plan, seqs):
        return []

    def allocate_kv_pool(self, num_blocks, block_size):
        self.kv_pool_calls.append((num_blocks, block_size))

    def capture_decode_graphs(self):
        self.graph_capture_calls += 1


@pytest.fixture(autouse=True)
def _clear_output_channels():
    output_channels.clear()
    yield
    output_channels.clear()


def make_engine() -> InferenceEngine:
    cfg = EngineConfig(model=ModelConfig(), kv_cache=KVCacheConfig(), server=ServerConfig())
    return InferenceEngine(config=cfg, tokenizer=FakeTokenizer(), model_runner=FakeModelRunner())


def test_submit_tokenizes_and_registers_output_channel():
    engine = make_engine()
    seq_id, q = engine.submit("hi", SamplingParams())
    assert isinstance(q, asyncio.Queue)
    assert output_channels[seq_id] is q


def test_submit_enqueues_ingress_request():
    engine = make_engine()
    seq_id, _ = engine.submit("hi", SamplingParams())
    req = engine.ingress.get_nowait()
    assert req.seq_id == seq_id
    assert req.prompt_tokens == [ord("h"), ord("i")]


def test_submit_bounds_the_output_queue_from_config():
    cfg = EngineConfig(
        model=ModelConfig(), kv_cache=KVCacheConfig(), server=ServerConfig(output_queue_maxsize=2)
    )
    engine = InferenceEngine(config=cfg, tokenizer=FakeTokenizer(), model_runner=FakeModelRunner())
    _, q = engine.submit("hi", SamplingParams())
    assert q.maxsize == 2


def test_submit_assigns_increasing_seq_ids():
    engine = make_engine()
    first, _ = engine.submit("a", SamplingParams())
    second, _ = engine.submit("b", SamplingParams())
    assert second == first + 1


def test_cancel_drops_output_channel():
    engine = make_engine()
    seq_id, _ = engine.submit("hi", SamplingParams())
    engine.cancel(seq_id)
    assert seq_id not in output_channels


def test_start_spawns_scheduler_and_worker_threads_and_stop_joins_them():
    engine = make_engine()
    engine.start()
    try:
        assert len(engine._threads) == 2
        assert all(t.is_alive() for t in engine._threads)
    finally:
        engine.stop()
    assert not any(t.is_alive() for t in engine._threads)


def test_submitted_request_reaches_running_via_the_scheduler_thread():
    engine = make_engine()
    engine.start()
    try:
        seq_id, _q = engine.submit("hi", SamplingParams())

        def admitted():
            return any(s.seq_id == seq_id for s in engine.running)

        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and not admitted():
            time.sleep(0.01)
        assert admitted()
    finally:
        engine.stop()


def test_scheduler_config_knob_selects_static_scheduler():
    cfg = EngineConfig(
        model=ModelConfig(), kv_cache=KVCacheConfig(), server=ServerConfig(),
        scheduler="static", static_batch_size=3,
    )
    engine = InferenceEngine(config=cfg, tokenizer=FakeTokenizer(), model_runner=FakeModelRunner())
    assert isinstance(engine.scheduler, StaticBatchedScheduler)
    assert engine.scheduler.batch_size == 3


def test_scheduler_config_knob_defaults_to_continuous():
    engine = make_engine()
    assert isinstance(engine.scheduler, ContinuousBatchedScheduler)


def test_max_concurrent_sequences_config_knob_reaches_the_continuous_scheduler():
    cfg = EngineConfig(
        model=ModelConfig(), kv_cache=KVCacheConfig(), server=ServerConfig(),
        max_concurrent_sequences=8,
    )
    engine = InferenceEngine(config=cfg, tokenizer=FakeTokenizer(), model_runner=FakeModelRunner())
    assert engine.scheduler.max_concurrent_sequences == 8


def test_allocator_config_knob_selects_contiguous_allocator_at_matched_capacity():
    cfg = EngineConfig(
        model=ModelConfig(), kv_cache=KVCacheConfig(block_size=16), server=ServerConfig(),
        kv_allocator="contiguous",
    )
    engine = InferenceEngine(config=cfg, tokenizer=FakeTokenizer(), model_runner=FakeModelRunner())
    assert isinstance(engine.allocator, ContiguousAllocator)
    paged_cfg = EngineConfig(
        model=ModelConfig(), kv_cache=KVCacheConfig(block_size=16), server=ServerConfig(),
    )
    paged_engine = InferenceEngine(
        config=paged_cfg, tokenizer=FakeTokenizer(), model_runner=FakeModelRunner()
    )
    assert isinstance(paged_engine.allocator, BlockAllocator)
    num_blocks = len(paged_engine.allocator.free_blocks)  # nothing allocated yet
    assert engine.allocator.capacity_tokens == num_blocks * 16


def test_paged_allocator_and_custom_kernels_allocates_the_kv_pool():
    cfg = EngineConfig(model=ModelConfig(), kv_cache=KVCacheConfig(block_size=16), server=ServerConfig())
    runner = FakeModelRunner()
    engine = InferenceEngine(config=cfg, tokenizer=FakeTokenizer(), model_runner=runner)
    num_blocks = len(engine.allocator.free_blocks)
    assert runner.kv_pool_calls == [(num_blocks, 16)]


def test_contiguous_allocator_skips_the_kv_pool():
    cfg = EngineConfig(
        model=ModelConfig(), kv_cache=KVCacheConfig(), server=ServerConfig(), kv_allocator="contiguous",
    )
    runner = FakeModelRunner()
    InferenceEngine(config=cfg, tokenizer=FakeTokenizer(), model_runner=runner)
    assert runner.kv_pool_calls == []


def test_non_custom_kernels_skips_the_kv_pool_even_when_paged():
    cfg = EngineConfig(
        model=ModelConfig(use_custom_kernels=False), kv_cache=KVCacheConfig(), server=ServerConfig(),
    )
    runner = FakeModelRunner()
    InferenceEngine(config=cfg, tokenizer=FakeTokenizer(), model_runner=runner)
    assert runner.kv_pool_calls == []


def test_use_cuda_graphs_default_on_captures_decode_graphs_after_the_kv_pool():
    cfg = EngineConfig(model=ModelConfig(), kv_cache=KVCacheConfig(), server=ServerConfig())
    runner = FakeModelRunner()
    InferenceEngine(config=cfg, tokenizer=FakeTokenizer(), model_runner=runner)
    assert runner.kv_pool_calls  # capture must come after the pool exists
    assert runner.graph_capture_calls == 1


def test_use_cuda_graphs_explicitly_off_skips_capture():
    cfg = EngineConfig(model=ModelConfig(use_cuda_graphs=False), kv_cache=KVCacheConfig(), server=ServerConfig())
    runner = FakeModelRunner()
    InferenceEngine(config=cfg, tokenizer=FakeTokenizer(), model_runner=runner)
    assert runner.graph_capture_calls == 0


def test_use_cuda_graphs_skipped_for_contiguous_allocator():
    cfg = EngineConfig(
        model=ModelConfig(use_cuda_graphs=True), kv_cache=KVCacheConfig(), server=ServerConfig(),
        kv_allocator="contiguous",
    )
    runner = FakeModelRunner()
    InferenceEngine(config=cfg, tokenizer=FakeTokenizer(), model_runner=runner)
    assert runner.graph_capture_calls == 0


def test_explicit_scheduler_and_allocator_override_the_config_knobs():
    cfg = EngineConfig(
        model=ModelConfig(), kv_cache=KVCacheConfig(), server=ServerConfig(), scheduler="static",
    )
    scheduler = ContinuousBatchedScheduler()
    allocator = ContiguousAllocator(capacity_tokens=100)
    engine = InferenceEngine(
        config=cfg,
        tokenizer=FakeTokenizer(),
        model_runner=FakeModelRunner(),
        scheduler=scheduler,
        allocator=allocator,
    )
    assert engine.scheduler is scheduler
    assert engine.allocator is allocator


def test_sequence_from_ingress_leaves_admission_to_the_scheduler():
    engine = make_engine()
    seq_id, _ = engine.submit("hi", SamplingParams())
    req = engine.ingress.get_nowait()
    seq = sequence_from_ingress(req)
    assert seq.seq_id == seq_id
    assert seq.status == "WAITING"
    assert seq.metrics.enqueue_time == req.arrival_time
    # Stamping admit_time here would hide the time spent in `waiting` from schedule_latency.
    assert seq.metrics.admit_time is None
