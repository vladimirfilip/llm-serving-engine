from llm_serving_engine.observability import metrics_export
from llm_serving_engine.observability.metrics import RequestMetrics
from llm_serving_engine.observability.metrics_export import (
    REGISTRY,
    generate_latest,
    record_request,
)


def _count(name: str, labels: dict[str, str] | None = None) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0


def test_record_request_observes_every_populated_stage():
    metrics = RequestMetrics(
        enqueue_time=0.0, admit_time=0.1, first_token_time=0.3, token_times=[0.3, 0.4, 0.6],
        done_time=1.0,
    )
    stages = ("schedule_latency", "prefill_latency", "inter_token_latency", "total_latency")
    before = {stage: _count(f"llm_{stage}_seconds_count") for stage in stages}

    record_request(metrics)

    observed = {stage: _count(f"llm_{stage}_seconds_count") - before[stage] for stage in stages}
    assert observed == {
        "schedule_latency": 1, "prefill_latency": 1, "inter_token_latency": 2, "total_latency": 1
    }


def test_record_request_skips_stages_that_never_happened():
    before = _count("llm_schedule_latency_seconds_count")
    record_request(RequestMetrics(enqueue_time=0.0))
    assert _count("llm_schedule_latency_seconds_count") == before


def test_record_request_labels_the_outcome_by_whether_the_request_completed():
    success = _count("llm_requests_total", {"outcome": "success"})
    error = _count("llm_requests_total", {"outcome": "error"})

    record_request(RequestMetrics(enqueue_time=0.0, done_time=1.0))
    record_request(RequestMetrics(enqueue_time=0.0))

    assert _count("llm_requests_total", {"outcome": "success"}) == success + 1
    assert _count("llm_requests_total", {"outcome": "error"}) == error + 1


def test_generate_latest_exports_this_registry():
    record_request(RequestMetrics(enqueue_time=0.0, done_time=0.5))
    text = generate_latest().decode()
    assert "llm_requests_total" in text
    assert "llm_total_latency_seconds_bucket" in text
    assert "llm_preemptions_total" in text
    assert metrics_export.CONTENT_TYPE_LATEST.startswith("text/plain")
