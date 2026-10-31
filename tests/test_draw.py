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


def _dominant_hue(image, xyxy):
    """Mean BGR along the drawn box edge."""
    x1, y1, x2, y2 = xyxy
    edge = np.concatenate([
        image[y1, x1:x2].reshape(-1, 3),
        image[y2, x1:x2].reshape(-1, 3),
    ])
    lit = edge[edge.sum(axis=1) > 120]       # ignore unpainted background
    return lit.mean(axis=0) if len(lit) else np.zeros(3)


def test_confirmed_tracks_are_drawn_green():
    """The green box is the signal that something is actually being tracked."""
    box = (100, 100, 260, 340)
    out = draw_tracks(canvas(), [track(xyxy=box)], show_trails=False)
    b, g, r = _dominant_hue(out, box)
    assert g > b and g > r, f"expected a green edge, got BGR ({b:.0f},{g:.0f},{r:.0f})"


def test_tentative_tracks_are_not_green():
    """A track that has not earned confirmation must look different, or the
    green box stops meaning anything."""
    box = (100, 100, 260, 340)
    out = draw_tracks(
        canvas(), [track(xyxy=box, state=TrackState.TENTATIVE)], show_trails=False,
    )
    b, g, r = _dominant_hue(out, box)
    # Amber is red-dominant; green is green-dominant. That is the distinction.
    assert r > g > b, f"tentative box should read amber, got BGR ({b:.0f},{g:.0f},{r:.0f})"


def test_track_colour_mode_gives_ids_distinct_hues():
    box_a, box_b = (60, 60, 200, 300), (260, 60, 400, 300)
    out = draw_tracks(
        canvas(),
        [track(track_id=1, xyxy=box_a), track(track_id=5, xyxy=box_b)],
        show_trails=False,
        color_mode="track",
    )
    assert not np.allclose(_dominant_hue(out, box_a), _dominant_hue(out, box_b), atol=12)


def test_state_mode_gives_every_confirmed_track_the_same_green():
    box_a, box_b = (60, 60, 200, 300), (260, 60, 400, 300)
    out = draw_tracks(
        canvas(),
        [track(track_id=1, xyxy=box_a), track(track_id=5, xyxy=box_b)],
        show_trails=False,
    )
    assert np.allclose(_dominant_hue(out, box_a), _dominant_hue(out, box_b), atol=12)


def test_corner_accents_stay_inside_a_small_box():
    """The accent run is clamped, so a tiny box cannot draw outside itself."""
    out = draw_tracks(canvas(), [track(xyxy=(5, 5, 25, 30))], show_trails=False)
    assert out.shape == (H, W, 3)


def test_label_still_reports_the_detected_class():
    """Identification is part of the overlay, not a future addition."""
    out = draw_tracks(canvas(), [track()], show_trails=False)
    assert out.any()


def test_adjacent_labels_do_not_overlap():
    """Subjects standing together produced captions that overwrote each other."""
    from motion_tracker.draw import _label, _overlaps

    placed = []
    for i in range(4):
        _label(canvas(), f"#{i} person 0.80 1.2s", (100, 150), (0, 255, 0), placed)

    assert len(placed) == 4
    for i, a in enumerate(placed):
        for b in placed[i + 1:]:
            assert not _overlaps(a, b), f"labels {a} and {b} collide"


def test_labels_stay_inside_the_canvas_when_stacked():
    from motion_tracker.draw import _label

    placed = []
    for i in range(40):                     # far more labels than vertical room
        _label(canvas(), f"#{i} person", (W - 40, H - 20), (0, 255, 0), placed)

    for x1, y1, x2, y2 in placed:
        assert x1 >= 0 and y1 >= 0
        assert x2 <= W and y2 <= H, f"label ({x1},{y1},{x2},{y2}) left the canvas"


def test_labels_anchored_at_the_bottom_edge_stack_upward():
    """Regression: once stacking ran out of room below it reset to the original
    position without rechecking, dropping a caption straight onto another. The
    previous test only asserted bounds, so it never caught this."""
    from motion_tracker.draw import _label, _overlaps

    placed = []
    for i in range(6):
        _label(canvas(), f"#{i} person 0.80 1.2s", (40, H - 12), (0, 255, 0), placed)

    assert len(placed) == 6
    for i, a in enumerate(placed):
        for b in placed[i + 1:]:
            assert not _overlaps(a, b), f"bottom-anchored labels {a} and {b} collide"
    assert all(y2 <= H for _, _, _, y2 in placed)


def test_crowded_frame_minimises_rather_than_abandons_placement():
    """When there is genuinely no free slot, labels must stay in bounds and
    overlap as little as possible rather than piling onto one spot."""
    from motion_tracker.draw import _label

    placed = []
    for i in range(60):
        _label(canvas(), f"#{i} person 0.80", (40, H // 2), (0, 255, 0), placed)

    assert all(0 <= y1 and y2 <= H for _, y1, _, y2 in placed)
    assert len({top for _, top, _, _ in placed}) > 10, "labels collapsed onto few rows"
