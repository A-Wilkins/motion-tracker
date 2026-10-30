"""Track lifecycle tests.

These are the rules that turn id'd boxes into claims about the world, so
they are the rules most worth pinning down.
"""

from motion_tracker.config import TrackConfig
from motion_tracker.detect import Detection
from motion_tracker.track import TrackManager, TrackState

FRAME = (720, 1280)  # (height, width)


def cfg(**kw):
    return TrackConfig(**{"min_hits": 3, "max_age": 5, "edge_margin": 8, **kw})


def det(track_id, xyxy, label="person", conf=0.9):
    return Detection(track_id, 0, label, conf, xyxy)


def feed(manager, detections, frames, start=0, t0=0.0, dt=1 / 30):
    """Push the same detections for N frames. Returns tracks closed en route."""
    closed = []
    for i in range(frames):
        closed += manager.update(detections, start + i, t0 + (start + i) * dt)
    return closed


def test_track_is_tentative_until_min_hits():
    manager = TrackManager(cfg(), FRAME)
    feed(manager, [det(1, (600, 300, 660, 450))], frames=2)
    assert manager.live[0].state is TrackState.TENTATIVE
    assert manager.confirmed == []

    feed(manager, [det(1, (600, 300, 660, 450))], frames=1, start=2)
    assert manager.live[0].state is TrackState.CONFIRMED
    assert len(manager.confirmed) == 1


def test_unconfirmed_track_is_discarded_not_reported():
    """A two-frame blip is noise; it must never surface as an event."""
    manager = TrackManager(cfg(), FRAME)
    feed(manager, [det(1, (600, 300, 660, 450))], frames=2)
    closed = feed(manager, [], frames=6, start=2)
    assert closed == []
    assert manager.live == []
    assert manager.total_confirmed == 0


def test_disappearing_at_border_is_an_exit():
    manager = TrackManager(cfg(), FRAME)
    feed(manager, [det(1, (1270, 300, 1279, 450))], frames=4)
    closed = feed(manager, [], frames=5, start=4)
    assert [t.state for t in closed] == [TrackState.EXITED]


def test_disappearing_mid_frame_is_lost_not_exited():
    manager = TrackManager(cfg(), FRAME)
    feed(manager, [det(1, (600, 300, 660, 450))], frames=4)
    closed = feed(manager, [], frames=5, start=4)
    assert [t.state for t in closed] == [TrackState.LOST]


def test_track_survives_a_gap_shorter_than_max_age():
    manager = TrackManager(cfg(), FRAME)
    feed(manager, [det(1, (600, 300, 660, 450))], frames=4)
    feed(manager, [], frames=3, start=4)            # occluded, under max_age
    assert manager.live and manager.live[0].age == 3

    feed(manager, [det(1, (610, 300, 670, 450))], frames=1, start=7)
    track = manager.live[0]
    assert track.age == 0 and track.state is TrackState.CONFIRMED
    assert track.hits == 5


def test_dwell_and_trail_accumulate():
    manager = TrackManager(cfg(), FRAME, trail_length=8)
    feed(manager, [det(1, (600, 300, 660, 450))], frames=10)
    track = manager.live[0]
    assert track.dwell_s == 9 / 30
    assert len(track.trail) == 8      # capped by trail_length
    assert track.hits == 10


def test_label_sticks_to_most_confident_observation():
    """Class flicker on an ambiguous subject must not rewrite the label."""
    manager = TrackManager(cfg(), FRAME)
    manager.update([det(1, (600, 300, 660, 450), "person", 0.92)], 0, 0.0)
    manager.update([det(1, (600, 300, 660, 450), "dog", 0.40)], 1, 0.03)
    assert manager.live[0].label == "person"

    manager.update([det(1, (600, 300, 660, 450), "dog", 0.97)], 2, 0.06)
    assert manager.live[0].label == "dog"


def test_detections_without_an_id_are_ignored():
    manager = TrackManager(cfg(), FRAME)
    manager.update([det(None, (600, 300, 660, 450))], 0, 0.0)
    assert manager.live == []


def test_velocity_is_signed_and_per_frame():
    manager = TrackManager(cfg(), FRAME)
    for i in range(5):
        manager.update([det(1, (600 + i * 10, 300, 660 + i * 10, 450))], i, i / 30)
    vx, vy = manager.live[0].velocity()
    assert round(vx, 3) == 10.0
    assert round(vy, 3) == 0.0
