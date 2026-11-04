"""Rendering tests.

No window is opened; these assert the overlay code cannot crash or write
outside the canvas, since a drawing bug would otherwise only appear the first
time someone runs it with a display attached.
"""

from collections import deque

import cv2
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


def test_recorder_writes_a_browser_playable_h264_file(tmp_path):
    """mp4v produces a valid file that Safari refuses to play, so the codec
    matters: the output must carry an avcC (H.264) box."""
    from motion_tracker.draw import Recorder

    path = tmp_path / "out.mp4"
    with Recorder(str(path), fps=30.0) as rec:
        for i in range(10):
            frame = canvas()
            frame[:, : 40 + i * 4] = 220          # something that actually changes
            rec.write(frame)

    assert rec.frames == 10
    assert path.exists() and path.stat().st_size > 0
    assert b"avcC" in path.read_bytes(), "output is not H.264"


def test_recorder_matches_the_real_frame_size(tmp_path):
    """The writer opens on the first frame, because a camera is free to ignore
    the resolution it was asked for."""
    from motion_tracker.draw import Recorder

    path = tmp_path / "sized.mp4"
    with Recorder(str(path)) as rec:
        rec.write(np.zeros((180, 320, 3), dtype=np.uint8))
        rec.write(np.zeros((180, 320, 3), dtype=np.uint8))

    cap = cv2.VideoCapture(str(path))
    try:
        assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == 320
        assert int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 180
    finally:
        cap.release()


def test_recorder_close_is_idempotent(tmp_path):
    from motion_tracker.draw import Recorder

    rec = Recorder(str(tmp_path / "x.mp4"))
    rec.write(canvas())
    rec.close()
    rec.close()          # must not raise


def _painted(image):
    """Count pixels that differ from the flat canvas background."""
    return int((np.abs(image.astype(int) - 40).max(axis=2) > 12).sum())


def test_stale_tracks_are_not_drawn():
    """Regression: a track survives a gap of up to max_age so identity can
    outlive an occlusion, but drawing its last box during that gap paints a
    rectangle on empty road and claims to see something that is not there."""
    stale = track()
    stale.age = 7                       # not seen on this frame

    out = draw_tracks(canvas(), [stale], show_trails=False)
    assert _painted(out) == 0, "a track not seen this frame must not be drawn"

    shown = draw_tracks(canvas(), [stale], show_trails=False, show_stale=True)
    assert _painted(shown) > 0, "show_stale should opt back in"


def test_visible_tracks_are_still_drawn():
    out = draw_tracks(canvas(), [track()], show_trails=False)
    assert _painted(out) > 0


def test_small_boxes_keep_their_outline_but_lose_the_caption():
    """In a busy scene the distant subjects are too small to label legibly and
    their captions bury the ones that matter."""
    tiny = track(xyxy=(100, 100, 108, 130))      # 8px wide of a 640px frame
    out = draw_tracks(canvas(), [tiny], show_trails=False)

    # the box itself is drawn
    assert _painted(out) > 0
    # but nothing is painted in the band a caption would occupy, kept clear of
    # the box outline's own antialiasing a few rows above y1
    assert _painted(out[70:94]) == 0, "a caption was drawn for a tiny box"


def test_large_boxes_keep_their_caption():
    big = track(xyxy=(100, 100, 300, 320))
    out = draw_tracks(canvas(), [big], show_trails=False)
    assert _painted(out[70:94]) > 0, "a legible box should still be captioned"


def _atom_order(path):
    import struct
    data = open(path, "rb").read()
    out, off = [], 0
    while off + 8 <= len(data):
        size = struct.unpack(">I", data[off:off + 4])[0]
        out.append(data[off + 4:off + 8].decode("latin1"))
        if size < 8:
            break
        off += size
    return out


def test_recorded_mp4_is_faststart_and_still_decodes(tmp_path):
    """OpenCV writes moov last, so a browser must fetch the end of the file
    before it can start playing. Relocating it shifts every chunk offset, so
    the real test is that the file still decodes afterwards."""
    from motion_tracker.draw import Recorder

    path = tmp_path / "fs.mp4"
    with Recorder(str(path), fps=30.0) as rec:
        for i in range(24):
            frame = canvas()
            frame[:, : 30 + i * 8] = 200
            rec.write(frame)

    order = _atom_order(path)
    assert "moov" in order and "mdat" in order
    assert order.index("moov") < order.index("mdat"), f"not faststart: {order}"

    cap = cv2.VideoCapture(str(path))
    try:
        decoded = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            decoded += 1
    finally:
        cap.release()
    assert decoded == 24, f"offsets broken: decoded {decoded} of 24 frames"


def test_faststart_leaves_a_non_mp4_alone():
    """Anything unexpected must leave the file untouched rather than corrupt it."""
    import tempfile, pathlib as pl
    from motion_tracker.draw import _faststart

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as fh:
        fh.write(b"this is not an mp4 at all")
        name = fh.name
    try:
        assert _faststart(name) is False
        assert pl.Path(name).read_bytes() == b"this is not an mp4 at all"
    finally:
        pl.Path(name).unlink()
