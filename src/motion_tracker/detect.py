"""YOLO detection with built-in multi-object tracking.

Ultralytics fuses detection and association in `model.track(persist=True)`,
so this module returns detections that already carry a tracker-assigned id.
Lifecycle logic (confirmation, dwell, exits) lives in track.py; this layer
only owns the model, the device, and coordinate sanity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from ultralytics import YOLO

from .config import DetectConfig


@dataclass(frozen=True)
class Detection:
    track_id: int | None       # None when the tracker did not assign an id
    class_id: int
    label: str
    confidence: float
    xyxy: tuple[int, int, int, int]

    @property
    def centroid(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.xyxy
        return (x1 + x2) // 2, (y1 + y2) // 2

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.xyxy
        return max(0, x2 - x1) * max(0, y2 - y1)


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "0"
    return "cpu"


class Detector:
    def __init__(self, cfg: DetectConfig, tracker: str) -> None:
        self._cfg = cfg
        self._tracker = tracker
        self.device = resolve_device(cfg.device)
        self.model = YOLO(cfg.weights)
        self.names: dict[int, str] = self.model.names

    def warmup(self, width: int, height: int) -> None:
        """Run one throwaway inference so the first real frame is not an outlier."""
        blank = np.zeros((height, width, 3), dtype=np.uint8)
        self.model.predict(
            blank, imgsz=self._cfg.imgsz, device=self.device, verbose=False,
        )

    def track(self, image: np.ndarray) -> list[Detection]:
        height, width = image.shape[:2]
        results = self.model.track(
            image,
            persist=True,              # carry tracker state across calls
            tracker=self._tracker,
            conf=self._cfg.conf,
            iou=self._cfg.iou,
            imgsz=self._cfg.imgsz,
            device=self.device,
            classes=self._cfg.classes or None,
            verbose=False,
        )
        if not results:
            return []

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)
        ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None

        detections: list[Detection] = []
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = _clamp(xyxy[i], width, height)
            if x2 <= x1 or y2 <= y1:
                continue
            class_id = int(class_ids[i])
            detections.append(Detection(
                track_id=int(ids[i]) if ids is not None else None,
                class_id=class_id,
                label=self.names.get(class_id, str(class_id)),
                confidence=float(confs[i]),
                xyxy=(x1, y1, x2, y2),
            ))
        return detections

    def reset_tracker(self) -> None:
        """Drop association state, e.g. after a long gated-off gap."""
        predictor = getattr(self.model, "predictor", None)
        for tracker in getattr(predictor, "trackers", None) or []:
            reset = getattr(tracker, "reset", None)
            if callable(reset):
                reset()


def _clamp(box: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    """Clip a box to the frame. Tracks are bound to the camera's field of view."""
    x1 = int(np.clip(box[0], 0, width - 1))
    y1 = int(np.clip(box[1], 0, height - 1))
    x2 = int(np.clip(box[2], 0, width - 1))
    y2 = int(np.clip(box[3], 0, height - 1))
    return x1, y1, x2, y2
