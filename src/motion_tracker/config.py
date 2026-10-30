"""Typed configuration loaded from config.yaml."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

import yaml


@dataclass
class CaptureConfig:
    source: int | str = 0
    width: int = 1280
    height: int = 720
    queue_size: int = 2
    # "auto" drops stale frames for live devices and keeps every frame for
    # file sources. Live capture must stay in the present; a file is a fixed
    # set of frames and silently discarding them would corrupt any benchmark.
    drop_stale: bool | str = "auto"
    # Mirror the feed. A front-facing camera is not mirrored by default, which
    # reads as wrong when you are the subject. Applied at capture so the boxes
    # match what is on screen.
    flip: bool = False


@dataclass
class MotionConfig:
    enabled: bool = True
    scale: float = 0.25
    history: int = 300
    var_threshold: int = 32
    min_area_frac: float = 0.0008
    dilate_iterations: int = 2
    cooldown_frames: int = 12


@dataclass
class DetectConfig:
    weights: str = "yolo11n.pt"
    conf: float = 0.35
    iou: float = 0.5
    imgsz: int = 640
    device: str = "auto"
    classes: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 5, 7, 15, 16])


@dataclass
class TrackConfig:
    tracker: str = "bytetrack.yaml"
    min_hits: int = 4
    max_age: int = 30
    edge_margin: int = 8


@dataclass
class DisplayConfig:
    enabled: bool = True
    show_motion_mask: bool = False
    trail_length: int = 48


@dataclass
class Config:
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    detect: DetectConfig = field(default_factory=DetectConfig)
    track: TrackConfig = field(default_factory=TrackConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        if path is None:
            return cls()
        raw = yaml.safe_load(Path(path).read_text()) or {}
        return _build(cls, raw)


def _build(cls: type, raw: dict[str, Any]) -> Any:
    """Construct a dataclass from a dict, recursing into nested dataclasses.

    Unknown keys raise rather than being silently dropped - a typo in
    config.yaml should be loud, not a mysteriously ignored setting.
    """
    # `from __future__ import annotations` makes field.type a string, so
    # resolve the real classes before deciding what to recurse into.
    hints = get_type_hints(cls)
    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown key(s) for {cls.__name__}: {sorted(unknown)}")

    kwargs: dict[str, Any] = {}
    for name, value in raw.items():
        ftype = hints[name]
        if isinstance(value, dict) and is_dataclass(ftype):
            kwargs[name] = _build(ftype, value)
        else:
            kwargs[name] = value
    return cls(**kwargs)
