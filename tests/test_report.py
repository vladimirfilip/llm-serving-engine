import pytest

from llm_serving_engine.loadgen.report import SLOThresholds, build_report, tpot


def _request(latency, first_token_latency, output_tokens, prompt_tokens=10, token_times=None):
    return {
        "success": True,
        "latency": latency,
        "first_token_latency": first_token_latency,
        "output_tokens": output_tokens,
        "prompt_tokens": prompt_tokens,
        "token_times": token_times or [],
    }


def test_tpot_needs_at_least_two_output_tokens():
    assert tpot(_request(1.0, 0.2, output_tokens=1)) is None
    assert tpot(_request(1.0, 0.2, output_tokens=0)) is None


def test_tpot_is_decode_time_per_additional_token():
    # 1.0s total, 0.2s to first token -> 0.8s decode over 4 remaining tokens
    assert tpot(_request(1.0, 0.2, output_tokens=5)) == pytest.approx(0.2)


def test_build_report_throughput_uses_wall_clock_duration_not_summed_latency():
    results = [_request(1.0, 0.1, output_tokens=3) for _ in range(10)]
    report = build_report(results, duration_s=5.0)
    assert report.throughput_req_s == pytest.approx(2.0)
    assert report.output_tokens_s == pytest.approx(30 / 5.0)
    assert report.input_tokens_s == pytest.approx(100 / 5.0)
    assert report.total_tokens_s == pytest.approx(130 / 5.0)


def test_build_report_skips_failed_requests():
    results = [_request(1.0, 0.1, output_tokens=3), {"success": False, "error": "boom"}]
    report = build_report(results, duration_s=1.0)
    assert report.throughput_req_s == pytest.approx(1.0)
    assert report.ttft.count == 1


def test_build_report_latency_summaries_reflect_named_metrics():
    results = [_request(1.0, 0.1, output_tokens=5), _request(2.0, 0.3, output_tokens=5)]
    report = build_report(results, duration_s=1.0)
    assert report.ttft.mean == pytest.approx((0.1 + 0.3) / 2)
    assert report.e2e_latency.mean == pytest.approx((1.0 + 2.0) / 2)
    assert report.tpot.mean == pytest.approx(((1.0 - 0.1) / 4 + (2.0 - 0.3) / 4) / 2)


def test_build_report_itl_flattens_token_times_across_requests():
    results = [
        _request(1.0, 0.1, output_tokens=3, token_times=[0.1, 0.2, 0.35]),
        _request(1.0, 0.1, output_tokens=2, token_times=[0.1, 0.3]),
    ]
    report = build_report(results, duration_s=1.0)
    assert report.itl.count == 3  # two gaps from the first request, one from the second


def test_build_report_no_slo_leaves_goodput_and_attainment_unset():
    report = build_report([_request(1.0, 0.1, output_tokens=3)], duration_s=1.0)
    assert report.goodput_req_s is None
    assert report.slo_attainment is None


def test_build_report_goodput_counts_only_requests_meeting_every_slo():
    results = [
        _request(latency=0.5, first_token_latency=0.05, output_tokens=5),  # meets both SLOs
        _request(latency=2.0, first_token_latency=0.05, output_tokens=5),  # e2e too slow
    ]
    slo = SLOThresholds(ttft_ms=100, e2e_ms=1000)
    report = build_report(results, duration_s=2.0, slo=slo)
    assert report.goodput_req_s == pytest.approx(0.5)
    assert report.slo_attainment["ttft"] == pytest.approx(100.0)
    assert report.slo_attainment["e2e"] == pytest.approx(50.0)
    assert report.slo_attainment["tpot"] is None  # no tpot_ms threshold configured
