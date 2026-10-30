"""Cheap motion gate that decides whether the detector should run at all.

Background subtraction on a quarter-resolution grayscale frame costs well
under a millisecond. YOLO costs tens of milliseconds. On a scene that is
static most of the time - which is every surveillance scene - gating the
expensive stage behind the cheap one is most of the performance story.

The gate answers "is anything moving?", not "where should I look?". Cropping
to motion regions before detection is a later optimisation; it complicates
coordinate mapping and hurts tracker continuity, so it is not done here.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import MotionConfig


@dataclass(frozen=True)
class MotionResult:
    active: bool          # should the detector run on this frame?
    moving: bool          # is motion present right now (ignoring cooldown)?
    area_frac: float      # fraction of the frame covered by motion blobs
    regions: list[tuple[int, int, int, int]]  # full-res xyxy boxes
    mask: np.ndarray | None  # low-res foreground mask, for debug overlay


class MotionGate:
    def __init__(self, cfg: MotionConfig) -> None:
        self._cfg = cfg
        self._subtractor = cv2.createBackgroundSubtractorMOG2(
            history=cfg.history,
            varThreshold=cfg.var_threshold,
            detectShadows=False,  # shadow labelling costs time we spend elsewhere
        )
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self._cooldown = 0
        self._warmup = 0

    def update(self, image: np.ndarray) -> MotionResult:
        if not self._cfg.enabled:
            return MotionResult(True, False, 0.0, [], None)

        height, width = image.shape[:2]
        small = cv2.resize(
            image, None,
            fx=self._cfg.scale, fy=self._cfg.scale,
            interpolation=cv2.INTER_AREA,
        )
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        mask = self._subtractor.apply(gray)
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        mask = cv2.dilate(mask, self._kernel, iterations=self._cfg.dilate_iterations)

        # The subtractor needs a few frames to learn a background; until then
        # every pixel reads as foreground. Run the detector during warmup so
        # we never miss a subject that was present from the first frame.
        self._warmup += 1
        if self._warmup <= 10:
            return MotionResult(True, False, 0.0, [], mask)

        small_area = mask.shape[0] * mask.shape[1]
        min_area = self._cfg.min_area_frac * small_area
        inv_scale = 1.0 / self._cfg.scale

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regions: list[tuple[int, int, int, int]] = []
        total = 0.0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            total += area
            x, y, w, h = cv2.boundingRect(contour)
            regions.append((
                max(0, int(x * inv_scale)),
                max(0, int(y * inv_scale)),
                min(width, int((x + w) * inv_scale)),
                min(height, int((y + h) * inv_scale)),
            ))

        moving = bool(regions)
        if moving:
            self._cooldown = self._cfg.cooldown_frames
        elif self._cooldown > 0:
            self._cooldown -= 1

        return MotionResult(
            active=moving or self._cooldown > 0,
            moving=moving,
            area_frac=total / small_area if small_area else 0.0,
            regions=regions,
            mask=mask,
        )
