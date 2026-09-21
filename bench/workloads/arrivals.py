"""Open-loop Poisson arrival schedules. Closed loops hide queueing collapse, so load sweeps
never use one."""

from __future__ import annotations

import math

import numpy as np


def request_count(
    rate: float, target_duration_s: float, min_requests: int, max_requests: int
) -> int:
    return max(min_requests, min(max_requests, math.ceil(rate * target_duration_s)))


def poisson_schedule(
    rate: float, n: int, seed: int, workload_index: int, rate_index: int, repeat: int
) -> np.ndarray:
    """Send times in seconds from the point's start. The same arguments give the same
    schedule, so every engine faces identical arrivals."""
    rng = np.random.default_rng([seed, workload_index, rate_index, repeat])
    return np.cumsum(rng.exponential(1 / rate, size=n))


def measurement_window(t_sched: np.ndarray, fractions: tuple[float, float]) -> tuple[float, float]:
    span = float(t_sched[-1])
    return fractions[0] * span, fractions[1] * span
