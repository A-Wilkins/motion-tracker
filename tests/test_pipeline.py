"""End-to-end test against the generated fixture clip.

Slow (it loads real weights and runs real inference), so it is marked and
skipped automatically when the fixture has not been generated.

  python scripts/make_test_clip.py
  pytest -m slow
"""

from pathlib import Path

import pytest

from motion_tracker.config import Config
from motion_tracker.pipeline import Pipeline
from motion_tracker.track import TrackState

CLIP = Path("assets/test_clip.mp4")

pytestmark = pytest.mark.skipif(
    not CLIP.exists(),
    reason="run scripts/make_test_clip.py to generate the fixture",
)


@pytest.fixture(scope="module")
def run():
    cfg = Config.load("config.yaml")
    cfg.capture.source = str(CLIP)
    cfg.display.enabled = False

    closed, snapshots = [], 0
    with Pipeline(cfg) as pipeline:
        for snapshot in pipeline.run():
            closed += snapshot.closed
            snapshots += 1
        return closed, snapshots, pipeline.metrics


@pytest.mark.slow
def test_every_frame_is_processed(run):
    """The file policy must not drop frames, or benchmarks are meaningless.

    Asserted exactly, not approximately: a loose bound here is what let the
    discarded probe frame hide.
    """
    import cv2

    capture = cv2.VideoCapture(str(CLIP))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()

    _, snapshots, metrics = run
    assert snapshots == total, f"processed {snapshots} of {total} frames"
    assert metrics.frames == snapshots


@pytest.mark.slow
def test_gate_skips_the_idle_stretches(run):
    _, _, metrics = run
    assert metrics.inferences < metrics.frames, "gate never skipped a frame"
    assert 0.1 < metrics.gate_skip_rate < 0.6


@pytest.mark.slow
def test_subject_is_detected_tracked_and_exits(run):
    closed, _, _ = run
    assert closed, "no tracks were confirmed and closed"
    assert any(t.label == "person" for t in closed)
    # The subject starts and ends off-canvas, so every track leaves at a border.
    assert all(t.state is TrackState.EXITED for t in closed), [
        (t.track_id, t.label, t.state) for t in closed
    ]


@pytest.mark.slow
def test_closed_tracks_carry_a_usable_history(run):
    closed, _, _ = run
    for track in closed:
        assert track.hits >= 4
        assert track.dwell_s > 0
        assert len(track.trail) >= 2
        assert track.last_frame > track.first_frame
