"""Track lifecycle on top of the association tracker.

The association tracker hands back ids per frame. That is not yet an answer
to "what happened". This layer turns a stream of id'd boxes into objects with
a history: when they appeared, how long they have been in view, where they
have travelled, whether they are still a candidate or a confirmed subject,
and whether they left through the edge of the frame or vanished mid-scene.

Everything here is bound to the camera's field of view. A track that reaches
the border and disappears is reported as having EXITED; one that disappears
in open frame is reported as LOST, which usually means occlusion or a
detection dropout rather than a real departure.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum

from .config import TrackConfig
from .detect import Detection


class TrackState(str, Enum):
    TENTATIVE = "tentative"   # seen, but not yet enough frames to trust
    CONFIRMED = "confirmed"   # survived min_hits, treat as real
    LOST = "lost"             # disappeared mid-frame; likely occluded
    EXITED = "exited"         # disappeared at the frame border; left the scene


@dataclass
class Track:
    track_id: int
    label: str
    class_id: int
    first_seen: float
    last_seen: float
    first_frame: int
    last_frame: int
    xyxy: tuple[int, int, int, int]
    confidence: float
    hits: int = 1
    age: int = 0                     # frames since last update
    state: TrackState = TrackState.TENTATIVE
    trail: deque[tuple[int, int]] = field(default_factory=lambda: deque(maxlen=64))

    @property
    def dwell_s(self) -> float:
        return self.last_seen - self.first_seen

    @property
    def centroid(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.xyxy
        return (x1 + x2) // 2, (y1 + y2) // 2

    @property
    def is_visible(self) -> bool:
        return self.age == 0

    def velocity(self) -> tuple[float, float]:
        """Mean px/frame displacement over the recent trail."""
        if len(self.trail) < 2:
            return 0.0, 0.0
        (x0, y0), (x1, y1) = self.trail[0], self.trail[-1]
        span = len(self.trail) - 1
        return (x1 - x0) / span, (y1 - y0) / span


class TrackManager:
    """Owns the set of live tracks and their transitions."""

    def __init__(self, cfg: TrackConfig, frame_size: tuple[int, int], trail_length: int = 64) -> None:
        self._cfg = cfg
        self._height, self._width = frame_size
        self._trail_length = trail_length
        self._tracks: dict[int, Track] = {}
        self._confirmed_ever: set[int] = set()

    # -- update -------------------------------------------------------

    def update(self, detections: list[Detection], frame_index: int, timestamp: float) -> list[Track]:
        """Fold this frame's detections in. Returns tracks closed by this call."""
        seen: set[int] = set()

        for detection in detections:
            if detection.track_id is None:
                continue  # unassociated box: real this frame, but not an object yet
            seen.add(detection.track_id)
            track = self._tracks.get(detection.track_id)
            if track is None:
                track = Track(
                    track_id=detection.track_id,
                    label=detection.label,
                    class_id=detection.class_id,
                    first_seen=timestamp,
                    last_seen=timestamp,
                    first_frame=frame_index,
                    last_frame=frame_index,
                    xyxy=detection.xyxy,
                    confidence=detection.confidence,
                    trail=deque(maxlen=self._trail_length),
                )
                self._tracks[detection.track_id] = track
            else:
                best_so_far = track.confidence
                track.last_seen = timestamp
                track.last_frame = frame_index
                track.xyxy = detection.xyxy
                track.confidence = detection.confidence
                track.hits += 1
                # Class can flip frame to frame on ambiguous subjects, so the
                # label sticks to the most confident observation yet seen
                # rather than following every wobble.
                if detection.confidence >= best_so_far:
                    track.label = detection.label
                    track.class_id = detection.class_id

            track.age = 0
            track.trail.append(detection.centroid)
            if track.hits >= self._cfg.min_hits and track.state is TrackState.TENTATIVE:
                track.state = TrackState.CONFIRMED
                self._confirmed_ever.add(track.track_id)

        closed: list[Track] = []
        for track_id, track in list(self._tracks.items()):
            if track_id in seen:
                continue
            track.age += 1
            if track.age < self._cfg.max_age:
                continue
            track.state = (
                TrackState.EXITED if self._at_border(track.xyxy) else TrackState.LOST
            )
            del self._tracks[track_id]
            # A track that never earned confirmation was noise, not an object.
            if track.track_id in self._confirmed_ever:
                closed.append(track)

        return closed

    def _at_border(self, xyxy: tuple[int, int, int, int]) -> bool:
        x1, y1, x2, y2 = xyxy
        margin = self._cfg.edge_margin
        return (
            x1 <= margin
            or y1 <= margin
            or x2 >= self._width - 1 - margin
            or y2 >= self._height - 1 - margin
        )

    # -- queries ------------------------------------------------------

    @property
    def live(self) -> list[Track]:
        return list(self._tracks.values())

    @property
    def confirmed(self) -> list[Track]:
        return [t for t in self._tracks.values() if t.state is TrackState.CONFIRMED]

    @property
    def visible(self) -> list[Track]:
        """Confirmed tracks updated on the most recent processed frame."""
        return [t for t in self.confirmed if t.is_visible]

    @property
    def total_confirmed(self) -> int:
        return len(self._confirmed_ever)
