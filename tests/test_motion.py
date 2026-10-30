import numpy as np

from motion_tracker.config import MotionConfig
from motion_tracker.motion import MotionGate

H, W = 480, 640
RNG = np.random.default_rng(0)


def still():
    """Flat grey with faint noise, as a real sensor would deliver."""
    base = np.full((H, W, 3), 96, dtype=np.uint8)
    return np.clip(base + RNG.normal(0, 2, base.shape), 0, 255).astype(np.uint8)


def with_block(x, size=140):
    frame = still()
    frame[180:180 + size, x:x + size] = 240
    return frame


def settle(gate, frames=40):
    """Run enough still frames for warmup and cooldown to expire."""
    result = None
    for _ in range(frames):
        result = gate.update(still())
    return result


def test_disabled_gate_always_runs_the_detector():
    gate = MotionGate(MotionConfig(enabled=False))
    result = gate.update(still())
    assert result.active and not result.moving


def test_detector_runs_during_warmup():
    """Until MOG2 has a background, every pixel reads as foreground, so the
    gate must stay open or a subject present from frame one is missed."""
    gate = MotionGate(MotionConfig())
    assert gate.update(still()).active


def test_static_scene_goes_idle():
    gate = MotionGate(MotionConfig())
    result = settle(gate)
    assert not result.moving
    assert not result.active
    assert result.regions == []


def test_movement_opens_the_gate_and_locates_itself():
    gate = MotionGate(MotionConfig())
    settle(gate)

    result = None
    for x in (100, 140, 180):
        result = gate.update(with_block(x))

    assert result.moving and result.active
    assert result.area_frac > 0
    assert result.regions, "a 140px block should produce at least one region"

    x1, y1, x2, y2 = result.regions[0]
    assert 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H
    assert x2 - x1 > 40, "region should be reported in full-resolution pixels"


def test_cooldown_keeps_the_gate_open_briefly_after_motion_stops():
    """A subject that pauses mid-frame should not instantly stop being tracked."""
    cfg = MotionConfig(cooldown_frames=10)
    gate = MotionGate(cfg)
    settle(gate)
    for x in (100, 140, 180):
        gate.update(with_block(x))

    first_still = gate.update(still())
    assert not first_still.moving
    assert first_still.active, "cooldown should hold the gate open"

    for _ in range(cfg.cooldown_frames + 2):
        result = gate.update(still())
    assert not result.active


def test_tiny_changes_are_filtered_by_min_area():
    gate = MotionGate(MotionConfig(min_area_frac=0.02))
    settle(gate)
    result = None
    for x in (100, 104, 108):
        result = gate.update(with_block(x, size=12))   # ~0.06% of the frame
    assert not result.moving
