import pytest

from llm_serving_engine.loadgen.report import (
    MIN_QUARTER_REQUESTS,
    SLOThresholds,
    build_point,
    build_report,
    tpot,
    ttft_growth,
    wall_clock_s,
)


def _request(latency, first_token_latency, output_tokens, completed_at=1.0, **fields):
    """An open-loop result; drop "scheduled_at" for a closed-loop one."""
    return {
        "scheduled_at": 0.0,
        "shape": "chat",
        "success": True,
        "latency": latency,
        "first_token_latency": first_token_latency,
        "output_tokens": output_tokens,
        "prompt_tokens": 10,
        "token_times": [],
        "completed_at": completed_at,
        **fields,
    }


def _closed_loop(result: dict) -> dict:
    return {k: v for k, v in result.items() if k != "scheduled_at"}


def _arrivals(ttfts: list[float], shape: str = "chat") -> list[dict]:
    """One open-loop request per second, with the given TTFTs in arrival order."""
    return [
        _request(ttft + 0.5, ttft, 3, scheduled_at=float(i), completed_at=i + ttft + 0.5,
                 shape=shape)
        for i, ttft in enumerate(ttfts)
    ]


def test_tpot_needs_at_least_two_output_tokens():
    assert tpot(_request(1.0, 0.2, output_tokens=1)) is None
    assert tpot(_request(1.0, 0.2, output_tokens=0)) is None


def test_tpot_is_decode_time_per_additional_token():
    # 0.8s of decode after the first token, over 4 further tokens
    assert tpot(_request(1.0, 0.2, output_tokens=5)) == pytest.approx(0.2)


def test_open_loop_rates_count_completions_inside_the_arrival_window():
    # Arrivals span 4s, so the steady window is [1s, 4s]: the 0.5s ramp-up is excluded, and
    # so is the 5s completion, which lands after the last arrival, while the backlog drains.
    results = [
        _request(1.0, 0.1, output_tokens=3, scheduled_at=min(t, 4.0), completed_at=t)
        for t in (0.5, 1.0, 2.0, 4.0, 5.0)
    ]
    report = build_report(results)
    assert report.wall_clock_s == 5.0
    assert report.throughput_req_s == pytest.approx(3 / 3.0)
    assert report.output_tokens_s == pytest.approx(9 / 3.0)
    assert report.total_tokens_s == pytest.approx(39 / 3.0)


def test_closed_loop_rates_count_completions_after_the_ramp_up():
    # No arrival schedule, so the window is the last completion, 5s, and the steady window
    # is [1.25s, 5s].
    results = [
        _closed_loop(_request(1.0, 0.1, output_tokens=3, completed_at=t))
        for t in (0.5, 1.0, 2.0, 4.0, 5.0)
    ]
    report = build_report(results)
    assert report.throughput_req_s == pytest.approx(3 / 3.75)


def test_open_loop_offered_rate_is_the_arrivals_actually_drawn():
    results = [_request(1.0, 0.1, output_tokens=3, scheduled_at=i * 0.375) for i in range(9)]
    assert build_report(results).offered_req_s == pytest.approx(3.0)


def test_closed_loop_results_have_no_offered_rate():
    results = [_closed_loop(_request(1.0, 0.1, output_tokens=3))]
    assert build_report(results).offered_req_s is None


def test_failed_requests_count_toward_the_wall_clock_but_not_throughput():
    results = [
        _request(1.0, 0.1, output_tokens=3, completed_at=1.0),
        {"scheduled_at": 2.0, "success": False, "error": "timeout", "latency": 0.5,
         "completed_at": 1.0},
    ]
    report = build_report(results)
    assert report.failures == 1
    assert report.wall_clock_s == 1.0
    assert report.throughput_req_s == pytest.approx(1 / 1.5)
    assert report.latency.ttft_by_shape["chat"].count == 1
    assert report.keeps_up is False


def test_latency_summaries_reflect_named_metrics():
    results = [_request(1.0, 0.1, output_tokens=5), _request(2.0, 0.3, output_tokens=5)]
    latency = build_report(results).latency
    assert latency.ttft_by_shape["chat"].mean == pytest.approx((0.1 + 0.3) / 2)
    assert latency.e2e_latency.mean == pytest.approx((1.0 + 2.0) / 2)
    assert latency.tpot.mean == pytest.approx(((1.0 - 0.1) / 4 + (2.0 - 0.3) / 4) / 2)


def test_ttft_and_e2e_are_summarized_separately_for_each_request_shape():
    results = [
        _request(1.0, 0.05, output_tokens=3),
        _request(1.2, 0.07, output_tokens=3),
        _request(9.0, 4.0, output_tokens=3, shape="chunked_document"),
    ]
    latency = build_report(results).latency
    assert set(latency.ttft_by_shape) == set(latency.e2e_by_shape) == {"chat", "chunked_document"}
    assert latency.ttft_by_shape["chat"].max == pytest.approx(0.07)
    assert latency.e2e_by_shape["chat"].max == pytest.approx(1.2)
    assert latency.ttft_by_shape["chunked_document"].p50 == pytest.approx(4.0)


def test_itl_flattens_token_gaps_across_requests():
    results = [
        _request(1.0, 0.1, output_tokens=3, token_times=[0.1, 0.2, 0.35]),
        _request(1.0, 0.1, output_tokens=2, token_times=[0.1, 0.3]),
    ]
    assert build_report(results).latency.itl.count == 3


def test_a_steady_open_loop_run_keeps_up():
    report = build_report(_arrivals([0.1] * 40))
    assert report.ttft_growth == pytest.approx(1.0)
    assert report.keeps_up is True


def test_a_growing_backlog_does_not_keep_up():
    report = build_report(_arrivals([0.1 * (i + 1) for i in range(40)]))
    assert report.ttft_growth > 2
    assert report.keeps_up is False


def test_too_few_requests_leave_the_keep_up_verdict_unknown():
    results = _arrivals([0.1, 5.0, 0.1, 5.0] * (MIN_QUARTER_REQUESTS - 1))
    assert ttft_growth(results) is None
    assert build_report(results).keeps_up is None


def test_ttft_growth_follows_the_most_common_shape_only():
    steady_chat = _arrivals([0.1] * 40)
    late_documents = [
        _request(9.0, 8.0, 3, shape="document", scheduled_at=39.0 + i / 10, completed_at=48.0)
        for i in range(5)
    ]
    report = build_report(steady_chat + late_documents)
    assert report.ttft_growth == pytest.approx(1.0)
    assert report.keeps_up is True


def test_closed_loop_results_have_no_keep_up_verdict_and_no_ttft():
    closed_loop = [_closed_loop(r) for r in _arrivals([0.1] * 40)]
    report = build_report(closed_loop)
    assert report.ttft_growth is None
    assert report.keeps_up is None
    assert report.latency.ttft_by_shape == {}


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
        _request(latency=2.0, first_token_latency=0.05, output_tokens=5, scheduled_at=2.0,
                 completed_at=2.0),
    ]
    report = build_report(results, slo=SLOThresholds(ttft_ms=100, e2e_ms=1000))
    assert report.goodput_req_s == pytest.approx(1 / 1.5)
    assert report.slo_attainment["ttft"] == pytest.approx(100.0)
    assert report.slo_attainment["e2e"] == pytest.approx(50.0)
    assert report.slo_attainment["tpot"] is None


def test_goodput_holds_a_closed_loop_result_to_every_slo_but_ttft():
    slow_first_token = _request(
        latency=0.5, first_token_latency=5.0, output_tokens=5, scheduled_at=1.0, completed_at=1.0
    )
    slo = SLOThresholds(ttft_ms=300, e2e_ms=1000)

    open_loop = build_report([slow_first_token], slo=slo)
    closed_loop = build_report([_closed_loop(slow_first_token)], slo=slo)

    assert open_loop.goodput_req_s == 0.0
    assert closed_loop.goodput_req_s == pytest.approx(1 / 0.75)


def test_a_point_reports_rates_per_repeat_and_pools_latencies():
    fast, slow = _arrivals([0.1] * 40), _arrivals([0.3] * 40)
    point = build_point([fast, slow])
    assert len(point.repeats) == 2
    assert point.latency.ttft_by_shape["chat"].count == 80
    assert point.latency.ttft_by_shape["chat"].max == pytest.approx(0.3)


def test_a_point_falls_behind_if_any_repeat_did_and_is_unknown_if_none_could_tell():
    steady, backlog = _arrivals([0.1] * 40), _arrivals([0.1 * (i + 1) for i in range(40)])
    short = _arrivals([0.1] * 8)
    assert build_point([steady, steady]).keeps_up is True
    assert build_point([steady, backlog]).keeps_up is False
    assert build_point([steady, short]).keeps_up is True
    assert build_point([short, short]).keeps_up is None
