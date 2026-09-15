import pytest

from llm_serving_engine.loadgen.report import (
    SLOThresholds,
    build_report,
    latency_growth,
    tpot,
    wall_clock_s,
)


def _request(latency, first_token_latency, output_tokens, completed_at=1.0, **fields):
    return {
        "success": True,
        "latency": latency,
        "first_token_latency": first_token_latency,
        "output_tokens": output_tokens,
        "prompt_tokens": 10,
        "token_times": [],
        "completed_at": completed_at,
        **fields,
    }


def test_tpot_needs_at_least_two_output_tokens():
    assert tpot(_request(1.0, 0.2, output_tokens=1)) is None
    assert tpot(_request(1.0, 0.2, output_tokens=0)) is None


def test_tpot_is_decode_time_per_additional_token():
    # 0.8s of decode after the first token, over 4 further tokens
    assert tpot(_request(1.0, 0.2, output_tokens=5)) == pytest.approx(0.2)


def test_throughput_divides_by_the_wall_clock_to_the_last_completion():
    results = [_request(1.0, 0.1, output_tokens=3, completed_at=t) for t in (1.0, 2.0, 5.0)]
    report = build_report(results)
    assert report.wall_clock_s == 5.0
    assert report.throughput_req_s == pytest.approx(3 / 5.0)
    assert report.output_tokens_s == pytest.approx(9 / 5.0)
    assert report.total_tokens_s == pytest.approx(39 / 5.0)


def test_failed_requests_count_toward_the_wall_clock_but_not_throughput():
    results = [
        _request(1.0, 0.1, output_tokens=3, completed_at=1.0),
        {"success": False, "error": "timeout", "latency": 60.0, "completed_at": 4.0},
    ]
    report = build_report(results)
    assert report.failures == 1
    assert report.throughput_req_s == pytest.approx(1 / 4.0)
    assert report.ttft.count == 1
    assert not report.keeps_up


def test_latency_summaries_reflect_named_metrics():
    results = [_request(1.0, 0.1, output_tokens=5), _request(2.0, 0.3, output_tokens=5)]
    report = build_report(results)
    assert report.ttft.mean == pytest.approx((0.1 + 0.3) / 2)
    assert report.e2e_latency.mean == pytest.approx((1.0 + 2.0) / 2)
    assert report.tpot.mean == pytest.approx(((1.0 - 0.1) / 4 + (2.0 - 0.3) / 4) / 2)


def test_itl_flattens_token_gaps_across_requests():
    results = [
        _request(1.0, 0.1, output_tokens=3, token_times=[0.1, 0.2, 0.35]),
        _request(1.0, 0.1, output_tokens=2, token_times=[0.1, 0.3]),
    ]
    assert build_report(results).itl.count == 3


def test_a_steady_open_loop_run_keeps_up():
    results = [_request(0.5, 0.1, 3, scheduled_at=float(i), completed_at=i + 0.5) for i in range(8)]
    report = build_report(results)
    assert report.latency_growth == pytest.approx(1.0)
    assert report.keeps_up


def test_a_growing_backlog_does_not_keep_up():
    results = [
        _request(0.5 * (i + 1), 0.1, 3, scheduled_at=float(i), completed_at=1.5 * i + 0.5)
        for i in range(8)
    ]
    assert latency_growth(results) > 2
    assert not build_report(results).keeps_up


def test_closed_loop_results_have_no_latency_growth():
    assert latency_growth([_request(0.5, 0.1, 3) for _ in range(8)]) is None


def test_wall_clock_of_no_results_is_zero():
    assert wall_clock_s([]) == 0.0
    assert build_report([]).throughput_req_s == 0.0


def test_no_slo_leaves_goodput_and_attainment_unset():
    report = build_report([_request(1.0, 0.1, output_tokens=3)])
    assert report.goodput_req_s is None
    assert report.slo_attainment is None


def test_goodput_counts_only_requests_meeting_every_slo():
    results = [
        _request(latency=0.5, first_token_latency=0.05, output_tokens=5, completed_at=1.0),
        _request(latency=2.0, first_token_latency=0.05, output_tokens=5, completed_at=2.0),
    ]
    report = build_report(results, slo=SLOThresholds(ttft_ms=100, e2e_ms=1000))
    assert report.goodput_req_s == pytest.approx(0.5)
    assert report.slo_attainment["ttft"] == pytest.approx(100.0)
    assert report.slo_attainment["e2e"] == pytest.approx(50.0)
    assert report.slo_attainment["tpot"] is None
