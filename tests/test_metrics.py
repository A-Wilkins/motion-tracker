"""Metrics tests."""

import time

from motion_tracker.metrics import Metrics, Stat, Timer


def test_throughput_excludes_startup_cost():
    """Regression: elapsed once started at construction, so model loading and
    GPU warmup were charged against the frame rate and understated it."""
    metrics = Metrics()
    time.sleep(0.25)              # stands in for loading weights
    assert metrics.elapsed_s == 0.0
    assert metrics.throughput == 0.0

    for _ in range(5):
        metrics.tick()
        time.sleep(0.01)

    assert metrics.elapsed_s < 0.2, "startup time leaked into elapsed"
    assert metrics.throughput > 25, f"throughput {metrics.throughput:.1f} too low"


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
