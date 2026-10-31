"""Overlay rendering.

Kept entirely separate from the pipeline so the tracker can run headless
without dragging any drawing code along.
"""

from __future__ import annotations

import colorsys

import cv2
import numpy as np

from .track import Track, TrackState

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_WHITE = (255, 255, 255)
_DIM = (150, 150, 150)

# BGR. Green reads as "locked on" and amber as "not yet trusted", which is the
# distinction that matters most when watching this live.
_GREEN = (64, 220, 64)
_AMBER = (60, 180, 235)


def color_for_id(track_id: int) -> tuple[int, int, int]:
    """Stable, well-spaced BGR colour per track id.

    The golden-ratio hue step keeps consecutive ids visually distinct instead
    of handing out three shades of the same green.
    """
    hue = (track_id * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def color_for_state(track: Track) -> tuple[int, int, int]:
    """Green once a track is confirmed, amber while it is still tentative."""
    return _GREEN if track.state is TrackState.CONFIRMED else _AMBER


def draw_tracks(
    image: np.ndarray,
    tracks: list[Track],
    show_trails: bool = True,
    color_mode: str = "state",
) -> np.ndarray:
    """Outline each track.

    `color_mode="state"` paints confirmed tracks green and tentative ones
    amber, which answers "is this thing actually being tracked?" at a glance.
    `color_mode="track"` gives every id its own hue instead, which is what you
    want when several subjects are in frame and you need to tell them apart.
    """
    placed: list[tuple[int, int, int, int]] = []
    for track in tracks:
        confirmed = track.state is TrackState.CONFIRMED
        if color_mode == "track":
            color = color_for_id(track.track_id) if confirmed else _DIM
        else:
            color = color_for_state(track)
        thickness = 2 if confirmed else 1
        x1, y1, x2, y2 = track.xyxy

        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
        if confirmed:
            _corners(image, track.xyxy, color)

        if show_trails and len(track.trail) > 1:
            points = np.array(track.trail, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(image, [points], False, color, 1, cv2.LINE_AA)
            cv2.circle(image, track.centroid, 3, color, -1, cv2.LINE_AA)

        caption = f"#{track.track_id} {track.label} {track.confidence:.2f}"
        if confirmed:
            caption += f" {track.dwell_s:.1f}s"
        _label(image, caption, (x1, y1), color, placed)
    return image


def _corners(image: np.ndarray, xyxy: tuple[int, int, int, int], color) -> None:
    """Thicken the box corners, so a confirmed lock reads clearly against a
    busy background where a one-pixel rectangle disappears."""
    x1, y1, x2, y2 = xyxy
    run = max(6, min(22, (x2 - x1) // 5, (y2 - y1) // 5))
    for cx, dx in ((x1, 1), (x2, -1)):
        for cy, dy in ((y1, 1), (y2, -1)):
            cv2.line(image, (cx, cy), (cx + dx * run, cy), color, 3, cv2.LINE_AA)
            cv2.line(image, (cx, cy), (cx, cy + dy * run), color, 3, cv2.LINE_AA)


def _label(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color,
    placed: list[tuple[int, int, int, int]] | None = None,
) -> None:
    """Draw text in a filled box, kept inside the frame and off other labels.

    Adjacent subjects produce adjacent boxes, so captions collide and overwrite
    each other. Each label is nudged down past any already drawn, which keeps
    every one readable when several people stand together.
    """
    x, y = origin
    (tw, th), baseline = cv2.getTextSize(text, _FONT, 0.45, 1)
    box_h = th + baseline + 4
    box_w = tw + 6

    top = y - box_h
    if top < 0:  # box would run off the top, so drop it inside the bbox
        top = y
    x = max(0, min(x, image.shape[1] - box_w))

    for _ in range(len(placed or ())):
        rect = (x, top, x + box_w, top + box_h)
        clash = next((r for r in (placed or ()) if _overlaps(rect, r)), None)
        if clash is None:
            break
        top = clash[3] + 2
        if top + box_h > image.shape[0]:   # ran out of room below
            top = max(0, y - box_h)
            break

    if placed is not None:
        placed.append((x, top, x + box_w, top + box_h))
    cv2.rectangle(image, (x, top), (x + box_w, top + box_h), color, cv2.FILLED)
    cv2.putText(image, text, (x + 3, top + th + 2), _FONT, 0.45, (0, 0, 0), 1, cv2.LINE_AA)


def _overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def draw_hud(image: np.ndarray, lines: list[str]) -> np.ndarray:
    """Translucent stats panel in the top-left corner."""
    if not lines:
        return image
    pad = 8
    widths, heights = [], []
    for line in lines:
        (tw, th), base = cv2.getTextSize(line, _FONT, 0.5, 1)
        widths.append(tw)
        heights.append(th + base + 4)
    panel_w = max(widths) + pad * 2
    panel_h = sum(heights) + pad * 2 - 4

    overlay = image[0:panel_h, 0:panel_w].copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (0, 0, 0), cv2.FILLED)
    cv2.addWeighted(overlay, 0.55, image[0:panel_h, 0:panel_w], 0.45,
                    0, image[0:panel_h, 0:panel_w])

    y = pad
    for line, line_h in zip(lines, heights):
        cv2.putText(image, line, (pad, y + line_h - 6), _FONT, 0.5, _WHITE, 1, cv2.LINE_AA)
        y += line_h
    return image


def draw_motion_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Tint moving pixels, for tuning the gate thresholds by eye."""
    if mask is None:
        return image
    full = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    tint = np.zeros_like(image)
    tint[:, :, 2] = full  # red channel
    return cv2.addWeighted(image, 1.0, tint, 0.35, 0)


def draw_frame_bounds(image: np.ndarray, margin: int) -> np.ndarray:
    """Outline the exit margin, i.e. the edge band that means 'left the scene'."""
    h, w = image.shape[:2]
    cv2.rectangle(image, (margin, margin), (w - margin, h - margin), (60, 60, 60), 1, cv2.LINE_AA)
    return image
