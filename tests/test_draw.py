"""Rendering tests.

No window is opened; these assert the overlay code cannot crash or write
outside the canvas, since a drawing bug would otherwise only appear the first
time someone runs it with a display attached.
"""

from collections import deque

import numpy as np

from motion_tracker.draw import (
    color_for_id, draw_frame_bounds, draw_hud, draw_motion_mask, draw_tracks,
)
from motion_tracker.track import Track, TrackState

H, W = 480, 640


def canvas():
    return np.full((H, W, 3), 40, dtype=np.uint8)


def track(track_id=1, xyxy=(100, 100, 200, 300), state=TrackState.CONFIRMED, trail=True):
    t = Track(
        track_id=track_id, label="person", class_id=0,
        first_seen=0.0, last_seen=1.5, first_frame=0, last_frame=45,
        xyxy=xyxy, confidence=0.87, hits=45, state=state,
        trail=deque([(150, 200), (160, 205), (170, 210)], maxlen=16) if trail else deque(maxlen=16),
    )
    return t


def test_colors_are_distinct_and_in_range():
    colors = {color_for_id(i) for i in range(24)}
    assert len(colors) == 24, "ids should not collide on the same colour"
    assert all(0 <= c <= 255 for color in colors for c in color)


def test_draws_tracks_without_mutating_shape():
    image = canvas()
    out = draw_tracks(image, [track(), track(2, state=TrackState.TENTATIVE)])
    assert out.shape == (H, W, 3) and out.dtype == np.uint8
    assert out.any()


def test_label_fits_when_box_touches_the_top_left_corner():
    """Caption origin at (0, 0) must not be drawn above or left of the canvas."""
    out = draw_tracks(canvas(), [track(xyxy=(0, 0, 60, 80))])
    assert out.shape == (H, W, 3)


def test_label_fits_when_box_touches_the_bottom_right_corner():
    out = draw_tracks(canvas(), [track(xyxy=(W - 30, H - 40, W - 1, H - 1))])
    assert out.shape == (H, W, 3)


def test_track_with_an_empty_trail_renders():
    out = draw_tracks(canvas(), [track(trail=False)])
    assert out.shape == (H, W, 3)


def test_hud_panel_renders_and_tolerates_no_lines():
    out = draw_hud(canvas(), ["30.0 fps   mps   yolo11n.pt", "gate ACTIVE"])
    assert out[0:4, 0:4].mean() < 40, "panel should darken the corner"
    assert draw_hud(canvas(), []).shape == (H, W, 3)


def test_motion_mask_upscales_from_gate_resolution():
    mask = np.zeros((H // 4, W // 4), dtype=np.uint8)
    mask[20:40, 20:40] = 255
    out = draw_motion_mask(canvas(), mask)
    assert out.shape == (H, W, 3)
    assert draw_motion_mask(canvas(), None).shape == (H, W, 3)


def test_frame_bounds_render():
    assert draw_frame_bounds(canvas(), 8).shape == (H, W, 3)
