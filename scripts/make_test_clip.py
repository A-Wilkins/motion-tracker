"""Generate a deterministic test clip so the pipeline can be exercised headless.

Structure of the clip, which is what makes it useful as a fixture:

  phase 1  empty scene      - the motion gate learns a background and idles
  phase 2  subject traverses - motion fires, detections associate into tracks
  phase 3  empty scene      - tracks age out and should close as EXITED

Usage: python scripts/make_test_clip.py [out.mp4]
"""

from __future__ import annotations

import shutil
import ssl
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

SUBJECT_PATH = Path("assets/subject.jpg")
# Ultralytics' public sample image. Fetched rather than vendored, so no
# third-party asset is redistributed in this repository.
SUBJECT_URL = "https://ultralytics.com/images/bus.jpg"

WIDTH, HEIGHT = 1280, 720
IDLE_HEAD, TRAVERSE, IDLE_TAIL = 15, 70, 45
SUBJECT_WIDTH = 340
RNG = np.random.default_rng(0)


def background() -> np.ndarray:
    """Flat grey with faint sensor noise, so MOG2 has something realistic."""
    base = np.full((HEIGHT, WIDTH, 3), 96, dtype=np.uint8)
    noise = RNG.normal(0, 2.0, base.shape)
    return np.clip(base.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _ssl_context() -> ssl.SSLContext:
    """Python installed from python.org ships no system CA bundle, so verification
    fails unless certifi's is supplied explicitly."""
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def fetch_subject() -> np.ndarray | None:
    """Download the sample subject on first run, then reuse it."""
    if not SUBJECT_PATH.exists():
        SUBJECT_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"fetching {SUBJECT_URL}")
        try:
            with urllib.request.urlopen(SUBJECT_URL, context=_ssl_context(), timeout=30) as response:
                with SUBJECT_PATH.open("wb") as handle:
                    shutil.copyfileobj(response, handle)
        except OSError as exc:
            SUBJECT_PATH.unlink(missing_ok=True)
            print(
                f"could not download the sample subject: {exc}\n"
                f"fetch it manually: curl -sSL -o {SUBJECT_PATH} {SUBJECT_URL}",
                file=sys.stderr,
            )
            return None
    return cv2.imread(str(SUBJECT_PATH))


def main() -> int:
    out_path = Path(sys.argv[1] if len(sys.argv) > 1 else "assets/test_clip.mp4")
    subject = fetch_subject()
    if subject is None:
        print(f"no usable subject image at {SUBJECT_PATH}", file=sys.stderr)
        return 1

    scale = SUBJECT_WIDTH / subject.shape[1]
    subject = cv2.resize(subject, (SUBJECT_WIDTH, int(subject.shape[0] * scale)))
    sh, sw = subject.shape[:2]
    top = (HEIGHT - sh) // 2

    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (WIDTH, HEIGHT),
    )
    if not writer.isOpened():
        print(f"could not open writer for {out_path}", file=sys.stderr)
        return 1

    # Travel far enough that the subject starts and ends off-frame, so the
    # border-exit path in TrackManager is actually taken.
    start_x, end_x = -sw, WIDTH

    for _ in range(IDLE_HEAD):
        writer.write(background())

    for i in range(TRAVERSE):
        frame = background()
        x = int(start_x + (end_x - start_x) * i / (TRAVERSE - 1))
        # Clip the paste region to the canvas; the subject is partly outside it.
        fx1, fx2 = max(0, x), min(WIDTH, x + sw)
        if fx2 > fx1:
            sx1, sx2 = fx1 - x, fx2 - x
            frame[top:top + sh, fx1:fx2] = subject[:, sx1:sx2]
        writer.write(frame)

    for _ in range(IDLE_TAIL):
        writer.write(background())

    writer.release()
    total = IDLE_HEAD + TRAVERSE + IDLE_TAIL
    print(f"wrote {out_path} ({total} frames, {WIDTH}x{HEIGHT})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
