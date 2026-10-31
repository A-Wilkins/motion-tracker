import time

import cv2
import numpy as np
import pytest

from motion_tracker.capture import CaptureError, VideoSource, _resolve_drop_policy
from motion_tracker.config import CaptureConfig


@pytest.mark.parametrize("source,expected", [
    (0, True),                              # webcam index: live
    ("1", True),                            # index as string: live
    ("rtsp://cam.local/stream", True),      # network stream: live
    ("assets/test_clip.mp4", False),        # file: keep every frame
    ("/abs/path/clip.mov", False),
])
def test_auto_policy_follows_source_kind(source, expected):
    assert _resolve_drop_policy(CaptureConfig(source=source)) is expected


def test_explicit_policy_wins():
    cfg = CaptureConfig(source="clip.mp4", drop_stale=True)
    assert _resolve_drop_policy(cfg) is True
    cfg = CaptureConfig(source=0, drop_stale=False)
    assert _resolve_drop_policy(cfg) is False


def _write_clip(path, frames=12, size=(160, 120)):
    """Clip whose nth frame is uniformly brighter than the (n-1)th, so frame
    order and completeness are both checkable after lossy encoding."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, size)
    assert writer.isOpened()
    for i in range(frames):
        # Step scaled to the frame count so brightness stays inside uint8
        # while remaining strictly increasing.
        level = 10 + round(i * 240 / max(1, frames - 1))
        writer.write(np.full((size[1], size[0], 3), level, dtype=np.uint8))
    writer.release()
    return frames


def test_file_source_yields_every_frame_including_the_first(tmp_path):
    """Regression: validating the device used to consume and discard frame 0,
    so a file source silently lost it and any frame-count benchmark was off."""
    path = tmp_path / "clip.mp4"
    expected = _write_clip(path, frames=12)

    with VideoSource(CaptureConfig(source=str(path))) as source:
        frames = list(source)

    assert len(frames) == expected
    assert [f.index for f in frames] == list(range(expected))
    assert source.frames_read == expected
    assert source.frames_dropped == 0

    brightness = [float(f.image.mean()) for f in frames]
    assert brightness == sorted(brightness), "frames arrived out of order"
    assert brightness[0] < 30, "the first frame of the clip is missing"


def test_failed_open_raises_with_a_useful_message(tmp_path):
    with pytest.raises(CaptureError, match="could not open"):
        VideoSource(CaptureConfig(source=str(tmp_path / "nope.mp4"))).open()


def test_flip_mirrors_frames_at_capture(tmp_path):
    """Mirroring must happen at capture, so detections line up with the view."""
    path = tmp_path / "sided.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (160, 120))
    assert writer.isOpened()
    for _ in range(4):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        frame[:, :40] = 255          # bright band on the left
        writer.write(frame)
    writer.release()

    def left_is_brighter(flip):
        cfg = CaptureConfig(source=str(path), flip=flip)
        with VideoSource(cfg) as source:
            image = next(iter(source)).image
        half = image.shape[1] // 2
        return bool(image[:, :half].mean() > image[:, half:].mean())

    assert left_is_brighter(flip=False), "unflipped frame lost its left-side band"
    assert not left_is_brighter(flip=True), "flip did not mirror the frame"


def test_recorded_timestamps_follow_video_time_not_decode_time(tmp_path):
    """Regression: the decoder runs ahead of the consumer, so wall-clock
    timestamps collapsed many frames into one instant and every dwell read
    as 0.0s. Recorded sources must use the video's presentation time."""
    path = tmp_path / "clip.mp4"
    frames = _write_clip(path, frames=60)          # 60 frames at 30fps = ~2s

    with VideoSource(CaptureConfig(source=str(path))) as source:
        captured = list(source)

    assert len(captured) == frames
    span = captured[-1].timestamp - captured[0].timestamp
    assert 1.8 < span < 2.2, f"video time span was {span:.3f}s, expected ~2s"

    # Monotonic and roughly evenly spaced, as video time should be.
    stamps = [f.timestamp for f in captured]
    assert stamps == sorted(stamps)
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    assert max(gaps) < 0.1, f"largest gap {max(gaps):.3f}s is not frame-paced"


def test_live_timestamps_use_the_wall_clock(tmp_path):
    """A live source is timed by when frames actually arrived."""
    path = tmp_path / "clip.mp4"
    _write_clip(path, frames=8)
    cfg = CaptureConfig(source=str(path), drop_stale=True)  # force live policy

    before = time.monotonic()
    with VideoSource(cfg) as source:
        first = next(iter(source))
    after = time.monotonic()

    assert before <= first.timestamp <= after
