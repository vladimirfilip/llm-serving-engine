import pytest

from llm_serving_engine.observability.metrics import (
    LatencySummary,
    RequestMetrics,
    percentile,
    summarize,
    summarize_stage_latencies,
)


def test_schedule_latency_none_until_admitted():
    m = RequestMetrics(enqueue_time=10.0)
    assert m.schedule_latency is None
    m.admit_time = 10.5
    assert m.schedule_latency == pytest.approx(0.5)


def test_prefill_latency_needs_admit_and_first_token():
    m = RequestMetrics(enqueue_time=0.0, admit_time=1.0)
    assert m.prefill_latency is None
    m.first_token_time = 1.2
    assert m.prefill_latency == pytest.approx(0.2)


def test_inter_token_latencies_are_consecutive_gaps():
    m = RequestMetrics(enqueue_time=0.0, token_times=[1.0, 1.1, 1.4])
    assert m.inter_token_latencies == pytest.approx([0.1, 0.3])


def test_inter_token_latencies_empty_for_fewer_than_two_tokens():
    assert RequestMetrics(enqueue_time=0.0, token_times=[1.0]).inter_token_latencies == []
    assert RequestMetrics(enqueue_time=0.0).inter_token_latencies == []


def test_total_latency_none_until_done():
    m = RequestMetrics(enqueue_time=5.0)
    assert m.total_latency is None
    m.done_time = 8.0
    assert m.total_latency == pytest.approx(3.0)


def test_percentile_empty_raises():
    with pytest.raises(ValueError):
        percentile([], 50)


def test_percentile_known_values():
    values = [1, 2, 3, 4, 5]
    assert percentile(values, 50) == 3
    assert percentile(values, 100) == 5
    assert percentile(values, 1) == 1


def test_percentile_rank_is_exact_where_float_multiplication_is_not():
    assert percentile(list(range(1000)), 99.9) == 998


def test_percentile_unsorted_input():
    assert percentile([5, 1, 3, 2, 4], 50) == 3


def test_summarize_empty_raises():
    with pytest.raises(ValueError):
        summarize([])


def test_summarize_known_values():
    s = summarize([1, 2, 3, 4, 5])
    assert isinstance(s, LatencySummary)
    assert s.count == 5
    assert s.min == 1
    assert s.max == 5
    assert s.mean == pytest.approx(3.0)
    assert s.p50 == 3
    assert s.p90 == 5
    assert s.p95 == 5


def test_summarize_stage_latencies_only_includes_populated_stages():
    metrics = [RequestMetrics(enqueue_time=0.0)]
    assert summarize_stage_latencies(metrics) == {}


def test_summarize_stage_latencies_aggregates_across_requests():
    metrics = [
        RequestMetrics(enqueue_time=0.0, admit_time=0.1, first_token_time=0.3, done_time=1.0),
        RequestMetrics(enqueue_time=0.0, admit_time=0.2, first_token_time=0.5, done_time=1.5),
    ]
    stages = summarize_stage_latencies(metrics)
    assert set(stages) == {"schedule_latency", "prefill_latency", "total_latency"}
    assert stages["schedule_latency"].count == 2
