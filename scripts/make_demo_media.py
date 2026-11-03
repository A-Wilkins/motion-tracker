"""Render demo media for the portfolio: an H.264 clip and still frames.

Runs the real pipeline over the fixture clip and captures the annotated view,
so what ships is literally what the tool draws rather than a mockup.

Usage: python scripts/make_demo_media.py [outdir]
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

from motion_tracker.config import Config
from motion_tracker.draw import (
    Recorder, draw_frame_bounds, draw_hud, draw_motion_mask, draw_tracks,
)
from motion_tracker.pipeline import Pipeline

CLIP = Path("assets/test_clip.mp4")
WEB_WIDTH = 960                      # plenty for a web page, roughly half the bytes
STILLS = {62: "overlay", 38: "early"}


def render(pipeline, snapshot, cfg, mask=False):
    canvas = snapshot.image.copy()
    if mask:
        canvas = draw_motion_mask(canvas, snapshot.motion.mask)
    canvas = draw_frame_bounds(canvas, cfg.track.edge_margin)
    canvas = draw_tracks(canvas, snapshot.tracks, color_mode=cfg.display.color_mode)
    return draw_hud(canvas, pipeline.hud_lines(snapshot))


def to_web(image):
    height, width = image.shape[:2]
    if width <= WEB_WIDTH:
        return image
    scale = WEB_WIDTH / width
    return cv2.resize(
        image, (WEB_WIDTH, round(height * scale)), interpolation=cv2.INTER_AREA,
    )


def main() -> int:
    outdir = Path(sys.argv[1] if len(sys.argv) > 1 else "assets/demo")
    outdir.mkdir(parents=True, exist_ok=True)

    if not CLIP.exists():
        print("run scripts/make_test_clip.py first", file=sys.stderr)
        return 1

    cfg = Config.load("config.yaml")
    cfg.capture.source = str(CLIP)
    cfg.display.enabled = False

    video = outdir / "motion-tracker-demo.mp4"
    stills: dict[str, object] = {}

    with Pipeline(cfg) as pipeline, Recorder(str(video)) as recorder:
        for snapshot in pipeline.run():
            frame = to_web(render(pipeline, snapshot, cfg))
            recorder.write(frame)
            name = STILLS.get(snapshot.frame.index)
            if name:
                stills[name] = frame.copy()
            if snapshot.frame.index == 62:
                stills["mask"] = to_web(render(pipeline, snapshot, cfg, mask=True))

    print(f"wrote {video} ({video.stat().st_size / 1024:.0f} KB, {recorder.frames} frames)")

    for name, image in stills.items():
        path = outdir / f"motion-tracker-{name}.webp"
        cv2.imwrite(str(path), image, [cv2.IMWRITE_WEBP_QUALITY, 88])
        print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
