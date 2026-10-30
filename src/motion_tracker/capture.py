"""Threaded frame capture.

Decoding runs on its own thread so inference never stalls the decoder.

Two backpressure policies, because live and recorded sources want opposite
things. For a camera, frames are perishable: when the pipeline falls behind,
stale frames are dropped rather than buffered, since a tracker fed old frames
reports the past confidently - worse than reporting the present at a lower
frame rate. For a file, frames are the dataset: the decoder blocks instead,
so a slow model costs wall-clock time rather than silently corrupting the
frame count any benchmark is computed from.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from .config import CaptureConfig


@dataclass(frozen=True)
class Frame:
    index: int
    timestamp: float  # seconds, monotonic
    image: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        """(height, width) of the frame."""
        return self.image.shape[0], self.image.shape[1]


class CaptureError(RuntimeError):
    pass


class VideoSource:
    """Background-threaded video source with drop-oldest backpressure."""

    def __init__(self, cfg: CaptureConfig) -> None:
        self._cfg = cfg
        self._drop_stale = _resolve_drop_policy(cfg)
        depth = max(1, cfg.queue_size) if self._drop_stale else 64
        self._queue: queue.Queue[Frame] = queue.Queue(maxsize=depth)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._capture: cv2.VideoCapture | None = None
        self._probe: Frame | None = None
        self._frames_read = 0
        self._frames_dropped = 0
        self._error: BaseException | None = None

    # -- lifecycle ----------------------------------------------------

    def open(self) -> "VideoSource":
        source = self._cfg.source
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise CaptureError(
                f"could not open video source {source!r}. If this is a webcam on "
                "macOS, check System Settings > Privacy & Security > Camera and "
                "grant access to your terminal."
            )

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cfg.height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._capture = cap

        # Validating the device costs one read. Keep that frame rather than
        # discarding it: for a file source it is frame 0 of the dataset, and
        # dropping it would break the very guarantee the file policy makes.
        ok, image = cap.read()
        if not ok:
            cap.release()
            raise CaptureError(
                f"opened video source {source!r} but the first read failed; the "
                "device may be in use by another application."
            )
        self._probe = Frame(0, time.monotonic(), image)
        self._frames_read = 1

        self._thread = threading.Thread(target=self._pump, name="capture", daemon=True)
        self._thread.start()
        return self

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self._probe = None

    def __enter__(self) -> "VideoSource":
        return self.open()

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- reader thread ------------------------------------------------

    def _pump(self) -> None:
        cap = self._capture
        assert cap is not None
        try:
            if self._probe is not None:
                probe, self._probe = self._probe, None
                self._offer(probe)
            while not self._stop.is_set():
                ok, image = cap.read()
                if not ok:
                    # End of file, or a transiently unhappy device.
                    self._stop.set()
                    break
                frame = Frame(self._frames_read, time.monotonic(), image)
                self._frames_read += 1
                self._offer(frame)
        except BaseException as exc:  # surfaced to the consumer thread
            self._error = exc
            self._stop.set()

    def _offer(self, frame: Frame) -> None:
        """Enqueue under the configured backpressure policy."""
        while not self._stop.is_set():
            try:
                self._queue.put_nowait(frame)
                return
            except queue.Full:
                if not self._drop_stale:
                    # Recorded source: wait for the consumer, lose nothing.
                    try:
                        self._queue.put(frame, timeout=0.25)
                        return
                    except queue.Full:
                        continue
                try:
                    self._queue.get_nowait()
                    self._frames_dropped += 1
                except queue.Empty:
                    pass

    # -- consumer API -------------------------------------------------

    def read(self, poll: float = 0.25) -> Frame | None:
        """Block for the next frame. Returns None once the source is done.

        None means "exhausted", never "slow" - the caller should not have to
        distinguish a stalled device from end-of-stream.
        """
        while True:
            if self._error is not None:
                raise CaptureError("capture thread failed") from self._error
            try:
                return self._queue.get(timeout=poll)
            except queue.Empty:
                if self._stop.is_set():
                    return None

    def __iter__(self):
        while (frame := self.read()) is not None:
            yield frame

    # -- stats --------------------------------------------------------

    @property
    def frames_read(self) -> int:
        return self._frames_read

    @property
    def frames_dropped(self) -> int:
        return self._frames_dropped

    @property
    def drops_stale_frames(self) -> bool:
        return self._drop_stale


def _resolve_drop_policy(cfg: CaptureConfig) -> bool:
    """Live sources drop stale frames; recorded ones keep every frame."""
    if cfg.drop_stale != "auto":
        return bool(cfg.drop_stale)
    source = cfg.source
    if isinstance(source, int):
        return True
    text = str(source)
    if text.isdigit():
        return True
    return text.startswith(("rtsp://", "rtmp://", "http://", "https://"))
