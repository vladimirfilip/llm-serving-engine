import numpy as np

from bench.workloads.arrivals import measurement_window, poisson_schedule, request_count


def test_mean_gap_matches_the_rate():
    schedule = poisson_schedule(rate=8.0, n=10_000, seed=1234, workload_index=0, rate_index=2,
                                repeat=0)
    assert abs(np.diff(schedule, prepend=0).mean() * 8.0 - 1) < 0.03


def test_same_seed_same_schedule_and_each_index_changes_it():
    base = poisson_schedule(4.0, 50, 1234, 1, 2, 0)
    assert (poisson_schedule(4.0, 50, 1234, 1, 2, 0) == base).all()
    for variant in ((1234, 0, 2, 0), (1234, 1, 3, 0), (1234, 1, 2, 1), (99, 1, 2, 0)):
        assert not np.array_equal(poisson_schedule(4.0, 50, *variant), base)


def test_request_count_is_clamped_between_floor_and_ceiling():
    assert request_count(0.5, 120, 200, 4000) == 200
    assert request_count(10, 120, 200, 4000) == 1200
    assert request_count(100, 120, 200, 4000) == 4000
    assert request_count(1.01, 120, 1, 4000) == 122


def test_window_is_a_fraction_of_the_arrival_span():
    schedule = np.array([1.0, 5.0, 10.0])
    assert measurement_window(schedule, (0.1, 0.9)) == (1.0, 9.0)
