import textwrap

import pytest

from motion_tracker.config import Config


def test_defaults_when_no_file():
    cfg = Config.load(None)
    assert cfg.capture.source == 0
    assert cfg.detect.weights.endswith(".pt")


def test_nested_overrides(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(textwrap.dedent("""
        capture:
          source: 2
          width: 640
        detect:
          conf: 0.5
    """))
    cfg = Config.load(path)
    assert cfg.capture.source == 2
    assert cfg.capture.width == 640
    assert cfg.capture.height == 720   # untouched default survives
    assert cfg.detect.conf == 0.5


def test_unknown_key_is_loud(tmp_path):
    """A typo in config.yaml must fail, not be silently ignored."""
    path = tmp_path / "config.yaml"
    path.write_text("detect:\n  confidence: 0.5\n")
    with pytest.raises(ValueError, match="confidence"):
        Config.load(path)


def test_shipped_config_parses():
    cfg = Config.load("config.yaml")
    assert cfg.track.min_hits > 0
    assert 0 < cfg.motion.scale <= 1
