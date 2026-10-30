"""Rolling performance counters.

Honest numbers for the README: wall-clock FPS, how much of each frame's
budget each stage consumed, and how often the motion gate saved an
inference. A portfolio project claiming real-time should be able to prove it.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class Stat:
    """Rolling mean over a fixed window, in milliseconds."""
    window: int = 60
    samples: deque[float] = field(default_factory=lambda: deque(maxlen=60))

    def add(self, ms: float) -> None:
        self.samples.append(ms)

    @property
    def mean(self) -> float:
        return sum(self.samples) / len(self.samples) if self.samples else 0.0

    @property
    def p95(self) -> float:
        if not self.samples:
            return 0.0
        ordered = sorted(self.samples)
        return ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]


class Timer:
    """Context manager that records elapsed wall time into a Stat."""

    def __init__(self, stat: Stat) -> None:
        self._stat = stat
        self._start = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stat.add((time.perf_counter() - self._start) * 1000.0)


class Metrics:
    def __init__(self, window: int = 60) -> None:
        self.motion = Stat(window, deque(maxlen=window))
        self.inference = Stat(window, deque(maxlen=window))
        self.total = Stat(window, deque(maxlen=window))
        self._frame_times: deque[float] = deque(maxlen=window)
        self.frames = 0
        self.inferences = 0
        self._started = time.perf_counter()

    def tick(self) -> None:
        self.frames += 1
        self._frame_times.append(time.perf_counter())

    @property
    def fps(self) -> float:
        if len(self._frame_times) < 2:
            return 0.0
        span = self._frame_times[-1] - self._frame_times[0]
        return (len(self._frame_times) - 1) / span if span > 0 else 0.0

    @property
    def gate_skip_rate(self) -> float:
        """Fraction of frames on which the detector did not have to run."""
        if not self.frames:
            return 0.0
        return 1.0 - (self.inferences / self.frames)

    @property
    def elapsed_s(self) -> float:
        return time.perf_counter() - self._started

    def summary(self) -> str:
        return (
            f"frames={self.frames} inferences={self.inferences} "
            f"gate_skip={self.gate_skip_rate * 100:.1f}% "
            f"fps={self.frames / self.elapsed_s:.1f} "
            f"motion={self.motion.mean:.2f}ms "
            f"infer={self.inference.mean:.1f}ms (p95 {self.inference.p95:.1f}ms)"
        )
