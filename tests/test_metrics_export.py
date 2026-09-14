import pytest

from llm_serving_engine.observability import metrics_export
from llm_serving_engine.observability.metrics import RequestMetrics
from llm_serving_engine.observability.metrics_export import (
    REGISTRY,
    generate_latest,
    record_request,
    sample_kv_utilization,
)


def test_record_request_observes_all_populated_stages():
    before = REGISTRY.get_sample_value(
        "llm_total_latency_seconds_count"
    ) or 0
    m = RequestMetrics(
        enqueue_time=0.0,
        admit_time=0.1,
        first_token_time=0.3,
        token_times=[0.3, 0.4, 0.6],
        done_time=1.0,
    )
    record_request(m)
    after = REGISTRY.get_sample_value("llm_total_latency_seconds_count")
    assert after == before + 1

    inter_token_count = REGISTRY.get_sample_value("llm_inter_token_latency_seconds_count")
    assert inter_token_count >= len(m.inter_token_latencies)


def test_record_request_skips_none_stages():
    before = REGISTRY.get_sample_value("llm_schedule_latency_seconds_count") or 0
    m = RequestMetrics(enqueue_time=0.0)  # nothing ever set beyond enqueue
    record_request(m)
    after = REGISTRY.get_sample_value("llm_schedule_latency_seconds_count")
    assert after == before  # schedule_latency was None, no observation recorded


def test_record_request_increments_outcome_counter():
    before_success = REGISTRY.get_sample_value(
        "llm_requests_total", {"outcome": "success"}
    ) or 0
    m = RequestMetrics(enqueue_time=0.0, done_time=1.0)
    record_request(m)
    after_success = REGISTRY.get_sample_value("llm_requests_total", {"outcome": "success"})
    assert after_success == before_success + 1


def test_generate_latest_produces_valid_prometheus_text():
    record_request(RequestMetrics(enqueue_time=0.0, done_time=0.5))
    text = generate_latest().decode()
    assert "llm_requests_total" in text
    assert "llm_total_latency_seconds_bucket" in text
    assert metrics_export.CONTENT_TYPE_LATEST.startswith("text/plain")


def test_in_flight_gauge_present_in_registry():
    metrics_export.REQUESTS_IN_FLIGHT.inc()
    assert REGISTRY.get_sample_value("llm_requests_in_flight") >= 1
    metrics_export.REQUESTS_IN_FLIGHT.dec()


class _BlockTable:
    def __init__(self, num_blocks):
        self.physical_blocks = list(range(num_blocks))


class _Seq:
    def __init__(self, num_blocks):
        self.block_table = _BlockTable(num_blocks)


class _BlockAllocator:
    def __init__(self, free_blocks):
        self.free_blocks = list(range(free_blocks))


class _ContiguousAllocator:
    def __init__(self, used_tokens, capacity_tokens):
        self.used_tokens = used_tokens
        self.capacity_tokens = capacity_tokens


class _FakeEngine:
    def __init__(self, allocator, running=()):
        self.allocator = allocator
        self.running = list(running)


def test_sample_kv_utilization_paged_allocator_uses_conserved_block_count():
    engine = _FakeEngine(_BlockAllocator(free_blocks=6), running=[_Seq(2), _Seq(2)])
    assert sample_kv_utilization(engine) == 4 / 10


def test_sample_kv_utilization_contiguous_allocator_uses_token_fraction():
    engine = _FakeEngine(_ContiguousAllocator(used_tokens=30, capacity_tokens=100))
    assert sample_kv_utilization(engine) == 0.3


def test_sample_kv_utilization_none_when_engine_has_no_allocator():
    class _NoAllocator:
        pass

    assert sample_kv_utilization(_NoAllocator()) is None


def test_kv_cache_utilization_gauge_is_settable_and_readable():
    metrics_export.KV_CACHE_UTILIZATION.set(0.42)
    assert REGISTRY.get_sample_value("llm_kv_cache_utilization") == pytest.approx(0.42)
