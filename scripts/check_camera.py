"""Probe camera indices and report which ones actually deliver frames.

On macOS, camera access is granted to the *application* that owns the shell,
not to the shell itself - so for an editor's integrated terminal the
responsible app is the editor. An app only appears under Privacy & Security >
Camera after it has requested access, and it can only request access if its
Info.plist declares NSCameraUsageDescription. Terminal.app does not declare
it, so a camera request from Terminal.app is denied without ever prompting.

This script makes that failure legible and names the app to look for.

Usage: python scripts/check_camera.py
"""

from __future__ import annotations

import os
import subprocess
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


def responsible_app() -> tuple[str | None, bool]:
    """Identify the app TCC will hold responsible, and whether it can prompt.

    Returns (bundle id, declares NSCameraUsageDescription).
    """
    bundle = os.environ.get("__CFBundleIdentifier")
    if not bundle or sys.platform != "darwin":
        return bundle, True
    try:
        path = subprocess.run(
            ["mdfind", f"kMDItemCFBundleIdentifier == {bundle}"],
            capture_output=True, text=True, timeout=10,
        ).stdout.splitlines()
        if not path:
            return bundle, True
        plist = f"{path[0]}/Contents/Info.plist"
        result = subprocess.run(
            ["/usr/libexec/PlistBuddy", "-c", "Print :NSCameraUsageDescription", plist],
            capture_output=True, text=True, timeout=10,
        )
        return bundle, result.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return bundle, True


def main() -> int:
    found = []
    for index in range(MAX_INDEX):
        ok, detail = probe(index)
        print(f"  index {index}: {'OK  ' if ok else 'FAIL'}  {detail}", flush=True)
        if ok:
            found.append(index)

    if found:
        print(f"\nusable camera indices: {found}")  # noqa: T201
        print(f"run: python -m motion_tracker --source {found[0]}")
        return 0

    sys.stdout.flush()
    print("\nNo camera opened. On macOS this is almost always permissions.", file=sys.stderr)

    bundle, can_prompt = responsible_app()
    if bundle and not can_prompt:
        print(
            f"\n  The app owning this shell ({bundle}) does not declare\n"
            "  NSCameraUsageDescription, so it cannot request camera access at all\n"
            "  and will never appear under Privacy & Security > Camera.\n"
            "  Terminal.app is one of these. Run this from an editor's integrated\n"
            "  terminal (VS Code, Cursor) or from iTerm2 instead.",
            file=sys.stderr,
        )
    else:
        print(
            f"\n  Run this directly in the foreground from the app owning your shell\n"
            f"  ({bundle or 'unknown'}) so macOS raises the permission prompt, then\n"
            "  click Allow. That app then appears under Privacy & Security > Camera.\n"
            "  If it was denied once before, clear the stale entry first:\n"
            f"    tccutil reset Camera {bundle or '<bundle-id>'}",
            file=sys.stderr,
        )

    print(
        "\nMeanwhile the fixture clip exercises everything but the camera:\n"
        "  python scripts/make_test_clip.py\n"
        "  python -m motion_tracker --source assets/test_clip.mp4",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
