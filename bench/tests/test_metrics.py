import math

import numpy as np
import pandas as pd
import pytest

from bench.metrics.aggregate import aggregate_repeats
from bench.metrics.goodput import Slo, max_sustainable_rps, meets_slo
from bench.metrics.request_metrics import derive, percentile
from bench.metrics.window import point_metrics, validity
from bench.tests.factories import frame, record

SLO = Slo(ttft_ms=200, tpot_ms=50)
WINDOW = (2.0, 8.0)


def steady(overrides: dict[int, dict] | None = None, n: int = 10) -> list[dict]:
    """One request per second at t = 0..n-1, five tokens each; `overrides[i]` edits request i."""
    return [record(i, float(i), **(overrides or {}).get(i, {})) for i in range(n)]


def test_latencies_derive_from_the_send_time_not_the_scheduled_time():
    df = derive(frame([record(0, 5.0, ttft=0.3, tpot=0.04, lag=0.5)]), itl_valid=True)
    assert df.send_lag[0] == pytest.approx(0.5)
    assert df.ttft[0] == pytest.approx(0.3)
    assert df.tpot[0] == pytest.approx(0.04)
    assert df.e2e[0] == pytest.approx(0.3 + 4 * 0.04)


def test_tpot_counts_usage_tokens_when_chunks_are_not_one_per_token():
    rec = record(0, 0.0, tpot=0.02, n_tokens=5)
    rec["n_chunks"] = 3  # the server bundled tokens
    rec["token_times"] = rec["token_times"][:3]
    bundled = derive(frame([rec]), itl_valid=False).tpot[0]
    assert bundled == pytest.approx(0.04 / 4)


def test_failed_requests_have_no_latencies_and_a_single_token_has_no_tpot():
    df = derive(frame([record(0, 0.0, status="error"), record(1, 1.0, n_tokens=1)]), True)
    assert math.isnan(df.ttft[0]) and math.isnan(df.tpot[1]) and df.ttft[1] == pytest.approx(0.1)


def test_percentile_uses_linear_interpolation_and_ignores_nan():
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([1, np.nan, 3], 50) == 2.0
    assert math.isnan(percentile([], 99))


def test_window_metrics_match_hand_computed_values():
    m = point_metrics(frame(steady()), WINDOW, SLO, itl_valid=True)
    # requests scheduled at t = 2..8 are in the window; 5 tokens each, 0.1 s to the first
    assert m["slo_attainment"] == 1.0
    assert m["goodput_rps"] == pytest.approx(7 / 6)
    # completions at k + 0.18 for k = 2..7 land in [2, 8]; request 8 finishes at 8.18
    assert m["achieved_rps"] == pytest.approx(6 / 6)
    # every token of requests 2..7 lies in the window: 30 tokens over 6 s
    assert m["out_tok_s"] == pytest.approx(5.0)
    assert m["ttft_p50"] == pytest.approx(0.1) and m["tpot_p99"] == pytest.approx(0.02)
    assert m["itl_p50"] == pytest.approx(0.02)
    assert m["error_rate"] == 0.0


def test_slow_requests_and_errors_count_as_misses_inside_the_window():
    records = steady({4: dict(tpot=0.06), 5: dict(ttft=0.5), 3: dict(status="error")})
    m = point_metrics(frame(records), WINDOW, SLO, itl_valid=True)
    assert m["slo_attainment"] == pytest.approx(4 / 7)
    assert m["goodput_rps"] == pytest.approx(4 / 6)
    assert m["error_rate"] == pytest.approx(1 / 10)


def test_requests_outside_the_window_do_not_count_toward_attainment():
    records = steady({0: dict(ttft=5.0), 9: dict(ttft=5.0)})
    assert point_metrics(frame(records), WINDOW, SLO, True)["slo_attainment"] == 1.0


def test_itl_is_nan_when_the_engine_does_not_stream_one_event_per_token():
    m = point_metrics(frame(steady()), WINDOW, SLO, itl_valid=False)
    assert math.isnan(m["itl_p50"]) and m["tpot_p50"] == pytest.approx(0.02)


def test_energy_per_token_divides_the_window_energy_by_its_tokens():
    m = point_metrics(frame(steady()), WINDOW, SLO, True, energy_j=60.0)
    assert m["energy_j_per_out_token"] == pytest.approx(60.0 / 30)


def test_meets_slo_bounds_are_inclusive_and_require_success():
    df = derive(frame([record(0, 0, ttft=0.2, tpot=0.05), record(1, 1, ttft=0.2001),
                       record(2, 2, status="timeout")]), True)
    assert meets_slo(df, SLO).tolist() == [True, False, False]


def test_max_sustainable_rate_stops_at_the_first_rate_that_fails():
    def points(pairs):
        return pd.DataFrame([{"offered_rps": r, "slo_attainment": a} for r, a in pairs])

    assert max_sustainable_rps(points([(1, 1.0), (2, 0.95), (3, 0.5), (4, 0.99)]), 0.9) == 2
    assert max_sustainable_rps(points([(1, 1.0), (2, 0.9)]), 0.9) == 2
    assert max_sustainable_rps(points([(1, 0.5), (2, 1.0)]), 0.9) is None


def test_max_sustainable_rate_uses_the_median_over_repeats():
    rows = [(1, 1.0), (1, 1.0), (1, 0.0), (2, 0.95), (2, 0.2), (2, 0.1)]
    pts = pd.DataFrame(rows, columns=["offered_rps", "slo_attainment"])
    assert max_sustainable_rps(pts, 0.9) == 1


def test_repeat_aggregation_takes_median_of_per_repeat_values_with_min_max():
    pts = pd.DataFrame({"offered_rps": [1, 1, 1, 2, 2], "ttft_p99": [0.3, 0.1, 0.2, 1.0, 2.0]})
    out = aggregate_repeats(pts, ["offered_rps"], ["ttft_p99"]).set_index("offered_rps")
    assert out.loc[1].tolist() == [0.2, 0.1, 0.3]
    assert out.loc[2].tolist() == [1.5, 1.0, 2.0]


def test_validity_names_every_reason_a_point_is_untrusted():
    ok = frame(steady())
    assert validity(ok, 5, 0.01, text_prompts=False, throttled=False) == []
    lagged = frame(steady({i: dict(lag=0.02) for i in range(10)}))
    assert "send_lag" in validity(lagged, 5, 0.01, False, False)[0]
    failing = frame(steady({0: dict(status="error"), 1: dict(status="error")}))
    assert "error rate" in validity(failing, 5, 0.01, False, False)
    assert "throttled" in validity(ok, 5, 0.01, False, True)
    short = ok.copy()
    short["completion_tokens_usage"] = 4
    assert any("completion_tokens" in r for r in validity(short, 5, 0.01, False, False))
    skewed = ok.copy()
    skewed["prompt_tokens_usage"] = skewed.prompt_len + 5
    assert validity(skewed, 5, 0.01, False, False) == []
    assert any("prompt_tokens" in r for r in validity(skewed, 5, 0.01, True, False))


def test_tokens_count_from_requests_that_arrived_earlier_or_later_failed():
    early = record(100, 1.0, ttft=0.5, tpot=0.5, n_tokens=5)  # tokens at 1.5, 2.0, 2.5, 3.0, 3.5
    timed_out = record(101, 7.0, ttft=0.5, tpot=0.5, n_tokens=5)
    timed_out.update(status="timeout", t_done=float("nan"), completion_tokens_usage=0)
    timed_out["token_times"] = timed_out["token_times"][:3]  # 7.5, 8.0 (edge), 8.5: two in [2, 8]
    m = point_metrics(frame([early, timed_out]), WINDOW, SLO, itl_valid=True)
    # early has 4 tokens in [2, 8] (2.0..3.5); the timed-out request delivered 2 before 8.0
    assert m["out_tok_s"] == pytest.approx((4 + 2) / 6)


def test_a_request_with_tokens_but_no_measurable_tpot_misses_the_slo():
    bundled = record(0, 0.0, n_tokens=5)
    bundled["n_chunks"], bundled["token_times"] = 1, bundled["token_times"][:1]
    df = derive(frame([bundled]), itl_valid=True)
    assert math.isnan(df.tpot[0]) and not meets_slo(df, SLO)[0]
