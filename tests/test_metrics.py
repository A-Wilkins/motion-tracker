"""Metrics tests."""

import time

import pytest

from motion_tracker.metrics import Metrics, Stat, Timer


class FakeClock:
    """Deterministic stand-in for perf_counter, so rates are exact."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_throughput_excludes_startup_cost(monkeypatch):
    """Regression: elapsed once started at construction, so model loading and
    GPU warmup were charged against the frame rate and understated it."""
    clock = FakeClock()
    monkeypatch.setattr("motion_tracker.metrics.time.perf_counter", clock)

    metrics = Metrics()
    clock.advance(2.5)            # stands in for loading weights and warmup
    assert metrics.elapsed_s == 0.0
    assert metrics.throughput == 0.0

    for _ in range(5):
        metrics.tick()
        clock.advance(0.01)

    assert metrics.elapsed_s == pytest.approx(0.05), "startup time leaked into elapsed"


def test_throughput_divides_by_intervals_not_frames(monkeypatch):
    """Regression: N frames span N-1 intervals, and the clock starts when the
    first frame finishes. Dividing by N inflated short runs badly."""
    clock = FakeClock()
    monkeypatch.setattr("motion_tracker.metrics.time.perf_counter", clock)

    metrics = Metrics()
    for i in range(5):
        metrics.tick()
        if i < 4:
            clock.advance(0.01)   # 4 intervals of 10ms between 5 frames

    # 4 intervals over 0.04s is exactly 100 fps. Dividing 5 frames by the same
    # span would read 125 - a 25% overstatement.
    assert metrics.throughput == pytest.approx(100.0)
    assert metrics.fps == pytest.approx(100.0), "fps and throughput should agree"


def test_a_single_frame_has_no_measurable_rate(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr("motion_tracker.metrics.time.perf_counter", clock)
    metrics = Metrics()
    metrics.tick()
    clock.advance(0.5)
    assert metrics.throughput == 0.0
    assert metrics.fps == 0.0


def test_gate_skip_rate_tracks_avoided_inferences():
    metrics = Metrics()
    for i in range(10):
        metrics.tick()
        if i % 2 == 0:
            metrics.inferences += 1
    assert metrics.gate_skip_rate == 0.5


def test_stat_mean_and_p95():
    stat = Stat(window=10)
    for value in range(1, 11):
        stat.add(float(value))
    assert stat.mean == 5.5
    assert stat.p95 == 10.0


def test_timer_records_into_its_stat():
    stat = Stat(window=4)
    with Timer(stat):
        time.sleep(0.02)
    assert 15 < stat.mean < 100


def test_empty_metrics_do_not_divide_by_zero():
    metrics = Metrics()
    assert metrics.fps == 0.0
    assert metrics.throughput == 0.0
    assert metrics.gate_skip_rate == 0.0
    assert "frames=0" in metrics.summary()
