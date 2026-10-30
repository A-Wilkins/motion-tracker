"""The pipeline: capture -> motion gate -> detect/track -> lifecycle.

Each processed frame yields a Snapshot. A consumer can render it, write it
to disk, push it to an event store, or ignore it entirely; the pipeline
itself has no opinion about output and no dependency on a display.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from .capture import Frame, VideoSource
from .config import Config
from .detect import Detection, Detector
from .metrics import Metrics, Timer
from .motion import MotionGate, MotionResult
from .track import Track, TrackManager


@dataclass
class Snapshot:
    frame: Frame
    motion: MotionResult
    detections: list[Detection]
    tracks: list[Track]          # live tracks, tentative and confirmed
    closed: list[Track]          # tracks that ended on this frame
    ran_inference: bool

    @property
    def image(self) -> np.ndarray:
        return self.frame.image


class Pipeline:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.metrics = Metrics()
        self._source = VideoSource(cfg.capture)
        self._gate = MotionGate(cfg.motion)
        self._detector = Detector(cfg.detect, cfg.track.tracker)
        self._tracks: TrackManager | None = None
        self._idle_frames = 0

    @property
    def detector(self) -> Detector:
        return self._detector

    @property
    def frames_dropped(self) -> int:
        """Frames the capture thread discarded to stay current (live sources)."""
        return self._source.frames_dropped

    def summary(self) -> str:
        """Run summary, including the live-source numbers metrics cannot see."""
        line = self.metrics.summary()
        dropped = self._source.frames_dropped
        if self._source.drops_stale_frames:
            total = self.metrics.frames + dropped
            share = dropped / total * 100 if total else 0.0
            line += f" dropped={dropped} ({share:.0f}% of captured)"
        return line

    @property
    def tracks(self) -> TrackManager:
        if self._tracks is None:
            raise RuntimeError("pipeline not started")
        return self._tracks

    def __enter__(self) -> "Pipeline":
        # Warm up before the camera starts producing. The first inference pays
        # for GPU kernel compilation, and anything the capture thread emits
        # during that window would be dropped unused.
        self._detector.warmup(self.cfg.capture.width, self.cfg.capture.height)
        self._source.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self._source.close()

    def run(self) -> Iterator[Snapshot]:
        while True:
            # Timed explicitly: time spent blocked here is the gap between
            # what the pipeline costs and what it actually delivers.
            with Timer(self.metrics.wait):
                frame = self._source.read()
            if frame is None:
                return
            height, width = frame.shape

            if self._tracks is None:
                self._tracks = TrackManager(
                    self.cfg.track, (height, width), self.cfg.display.trail_length,
                )
            with Timer(self.metrics.total):
                with Timer(self.metrics.motion):
                    motion = self._gate.update(frame.image)

                detections: list[Detection] = []
                if motion.active:
                    with Timer(self.metrics.inference):
                        detections = self._detector.track(frame.image)
                    self.metrics.inferences += 1
                    self._idle_frames = 0
                else:
                    self._idle_frames += 1
                    # After a long quiet stretch, stale Kalman predictions are
                    # worthless and recycled ids would be misleading. Start clean.
                    if self._idle_frames == self.cfg.track.max_age * 4:
                        self._detector.reset_tracker()

                closed = self._tracks.update(detections, frame.index, frame.timestamp)

            self.metrics.tick()
            yield Snapshot(
                frame=frame,
                motion=motion,
                detections=detections,
                tracks=self._tracks.live,
                closed=closed,
                ran_inference=motion.active,
            )

    # -- convenience --------------------------------------------------

    def hud_lines(self, snapshot: Snapshot) -> list[str]:
        gate = "ACTIVE" if snapshot.ran_inference else "idle"
        return [
            f"{self.metrics.fps:5.1f} fps   {self._detector.device}   {self.cfg.detect.weights}",
            f"gate {gate}   motion {snapshot.motion.area_frac * 100:4.1f}%   "
            f"skipped {self.metrics.gate_skip_rate * 100:.0f}%",
            f"infer {self.metrics.inference.mean:5.1f}ms   "
            f"tracks {len(self.tracks.visible)} live / {self.tracks.total_confirmed} total",
            f"dropped frames {self._source.frames_dropped}",
        ]
