"""Entry point: python -m motion_tracker [--config config.yaml]"""

from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path

import cv2

from .config import Config
from .draw import draw_frame_bounds, draw_hud, draw_motion_mask, draw_tracks
from .pipeline import Pipeline
from .track import TrackState

WINDOW = "motion tracker"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="motion_tracker", description=__doc__)
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    parser.add_argument("--source", help="override capture source (webcam index or path)")
    parser.add_argument("--weights", help="override detector weights")
    parser.add_argument("--no-display", action="store_true", help="run headless")
    parser.add_argument("--no-gate", action="store_true", help="disable the motion gate")
    parser.add_argument("--mask", action="store_true", help="overlay the motion mask")
    parser.add_argument("--flip", action="store_true", help="mirror the feed (front-facing camera)")
    parser.add_argument(
        "--color", choices=("state", "track"), default=None,
        help="box colour: 'state' green when confirmed, 'track' a hue per id",
    )
    parser.add_argument("--max-frames", type=int, help="stop after N frames (for benchmarking)")
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> Config:
    path = Path(args.config)
    cfg = Config.load(path if path.exists() else None)
    if args.source is not None:
        cfg.capture.source = int(args.source) if args.source.isdigit() else args.source
    if args.weights:
        cfg.detect.weights = args.weights
    if args.no_display:
        cfg.display.enabled = False
    if args.no_gate:
        cfg.motion.enabled = False
    if args.mask:
        cfg.display.show_motion_mask = True
    if args.flip:
        cfg.capture.flip = True
    if args.color:
        cfg.display.color_mode = args.color
    return cfg


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = build_config(args)

    # Ctrl-C should print the run summary, not a traceback.
    signal.signal(signal.SIGINT, signal.default_int_handler)

    show = cfg.display.enabled
    if show:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, cfg.capture.width, cfg.capture.height)

    try:
        with Pipeline(cfg) as pipeline:
            print(f"device={pipeline.detector.device} weights={cfg.detect.weights}", file=sys.stderr)
            print("q or esc to quit", file=sys.stderr)

            for snapshot in pipeline.run():
                for track in snapshot.closed:
                    verb = "exited" if track.state is TrackState.EXITED else "lost"
                    print(
                        f"[track {track.track_id:>3}] {track.label:<10} {verb:<6} "
                        f"dwell={track.dwell_s:5.1f}s hits={track.hits}",
                        flush=True,
                    )

                if show:
                    canvas = snapshot.image.copy()
                    if cfg.display.show_motion_mask:
                        canvas = draw_motion_mask(canvas, snapshot.motion.mask)
                    canvas = draw_frame_bounds(canvas, cfg.track.edge_margin)
                    canvas = draw_tracks(
                        canvas, snapshot.tracks, color_mode=cfg.display.color_mode,
                    )
                    canvas = draw_hud(canvas, pipeline.hud_lines(snapshot))
                    cv2.imshow(WINDOW, canvas)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break

                if args.max_frames and pipeline.metrics.frames >= args.max_frames:
                    break

            print(pipeline.summary(), file=sys.stderr)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
    finally:
        if show:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
