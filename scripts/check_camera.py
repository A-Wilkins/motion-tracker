"""Probe camera indices and report which ones actually deliver frames.

On macOS, camera access is gated by TCC per *application*, so a terminal
that has never been granted access fails to open the device rather than
prompting. This script makes that failure legible.

Usage: python scripts/check_camera.py
"""

from __future__ import annotations

import sys

import cv2

MAX_INDEX = 4


def probe(index: int) -> tuple[bool, str]:
    capture = cv2.VideoCapture(index)
    try:
        if not capture.isOpened():
            return False, "could not open"
        ok, frame = capture.read()
        if not ok or frame is None:
            return False, "opened but no frames (device busy?)"
        h, w = frame.shape[:2]
        fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        return True, f"{w}x{h} @ {fps:.0f}fps"
    finally:
        capture.release()


def main() -> int:
    found = []
    for index in range(MAX_INDEX):
        ok, detail = probe(index)
        print(f"  index {index}: {'OK  ' if ok else 'FAIL'}  {detail}")
        if ok:
            found.append(index)

    if found:
        print(f"\nusable camera indices: {found}")
        print(f"run: python -m motion_tracker --source {found[0]}")
        return 0

    print(
        "\nNo camera opened. On macOS this is almost always permissions:\n"
        "  System Settings > Privacy & Security > Camera\n"
        "  enable the app you are running this from (Terminal, iTerm, VS Code).\n"
        "You may need to fully quit and reopen that app afterwards.\n"
        "Meanwhile you can run against the test clip:\n"
        "  python scripts/make_test_clip.py\n"
        "  python -m motion_tracker --source assets/test_clip.mp4",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
