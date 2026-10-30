# motion-tracker

Real-time motion detection, object identification and multi-object tracking on a
live webcam feed. Python, OpenCV and YOLO11, running on Apple silicon via MPS.

Motion is detected with background subtraction, confirmed by an object detector,
and associated across frames into tracks that carry a history: class, confidence,
dwell time, trajectory, and whether the subject left the frame or was lost to
occlusion. Tracking is bound to the camera's field of view.

```bash
python -m motion_tracker                      # live webcam
python -m motion_tracker --source clip.mp4    # recorded file
python -m motion_tracker --mask               # visualise the motion gate
python -m motion_tracker --no-display         # headless, prints track events
```

## Design

```
capture ──▶ motion gate ──▶ detect + associate ──▶ lifecycle ──▶ snapshot
(thread)     ~1.3 ms          ~25 ms                              per frame
```

**Capture is threaded, with a backpressure policy that depends on the source.**
A camera's frames are perishable: when the pipeline falls behind, stale frames
are dropped, because a tracker fed old frames reports the past confidently. A
file's frames are the dataset: the decoder blocks instead, so a slow model costs
wall-clock time rather than silently corrupting the frame count a benchmark is
computed from. The policy is chosen from the source kind and can be overridden.

**The motion gate decides whether the detector runs at all.** Background
subtraction on a quarter-resolution grayscale frame costs ~1.3 ms; YOLO costs
~25 ms. Every surveillance scene is static most of the time, so gating the
expensive stage behind the cheap one is most of the performance story. The gate
answers "is anything moving?", not "where should I look?" — cropping to motion
regions before detection complicates coordinate mapping and hurts tracker
continuity, so it is deliberately not done.

Two details that matter more than they look: the gate holds open for a cooldown
window after motion stops, so a subject who pauses is not instantly dropped; and
it stays open during MOG2's warmup, when every pixel reads as foreground, so a
subject present from the first frame is never missed.

**Detection and association** use `YOLO.track(persist=True)` with ByteTrack.
Boxes are clipped to the frame — tracks cannot describe anything the camera
cannot see.

**The lifecycle layer is what makes this a system rather than a box-drawer.**
Per-frame ids are not yet an answer to "what happened", so tracks are promoted
through explicit states:

| state | meaning |
|---|---|
| `TENTATIVE` | seen, but has not survived `min_hits` frames yet |
| `CONFIRMED` | survived confirmation, treat as a real subject |
| `EXITED` | disappeared within `edge_margin` of the border — left the scene |
| `LOST` | disappeared in open frame — occlusion or a detection dropout |

Separating `EXITED` from `LOST` is the difference between "someone walked away"
and "I stopped being able to see them", which are not the same event. Tracks
that never reach `CONFIRMED` are discarded silently: a two-frame blip is noise,
and a surveillance system that reports noise as events is worse than useless.

## Measured

M1 MacBook, 1280x720, YOLO11n on MPS, over the 130-frame fixture clip. Median of
three runs, range in brackets — a laptop under thermal management is noisy
enough that a single run should not be quoted.

| | gate on | gate off |
|---|---|---|
| frames processed | 130 | 130 |
| inferences run | 89 | 130 |
| frames skipped | 31.5% | 0% |
| end-to-end | **33.2 fps** [26.6–35.5] | 27.2 fps [26.3–29.3] |
| motion gate | 1.3 ms | — |
| inference, mean | 24.7 ms | 21.4 ms |

Frame and inference counts are deterministic; throughput is not. Call it a ~1.2x
median speedup, which is about what skipping 31.5% of an inference-dominated
pipeline should buy.

Note the inversion in the last row: mean inference is *higher* with the gate on.
That is not noise. Gating means the detector only ever sees frames that contain
something, while the ungated run dilutes its average with empty frames that are
cheap to post-process. The gate's win is in how many inferences it avoids, not
in making any one of them faster — and a benchmark that only reported mean
inference latency would have shown the gate making things worse.

The fixture is idle for roughly a third of its length and the gate recovers
almost exactly that. On a real scene — a doorway, a driveway — idle time is the
overwhelming majority of the footage and the saving is correspondingly larger.
Reproduce with `python -m motion_tracker --source assets/test_clip.mp4
--no-display [--no-gate]`.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"                 # drop [dev] to skip the test deps
python scripts/check_camera.py          # verify camera access
python -m motion_tracker
```

On macOS, camera access is granted per application. If `check_camera.py` reports
no usable device, enable your terminal under System Settings → Privacy &
Security → Camera, then fully quit and reopen it.

## Tests

```bash
pytest              # 35 unit tests, no model weights needed
pytest -m slow      # 4 end-to-end tests against the fixture clip
```

The fixture is generated, not committed: `python scripts/make_test_clip.py`
builds a clip that is idle, then has a subject traverse the frame and leave at
the border, then is idle again — which exercises gating, confirmation and
border-exit in one pass. It downloads a public sample image on first run rather
than vendoring one into the repo.

## Configuration

Every tunable lives in `config.yaml` — capture resolution, gate sensitivity and
cooldown, detector weights and confidence, the class whitelist, confirmation and
age thresholds, and the border margin. CLI flags override the common ones.

## Layout

```
src/motion_tracker/
  capture.py    threaded video source, source-aware backpressure
  motion.py     MOG2 motion gate
  detect.py     YOLO wrapper, device selection, box clipping
  track.py      track lifecycle and state transitions
  pipeline.py   stage wiring, emits one Snapshot per frame
  metrics.py    rolling FPS and per-stage latency
  draw.py       overlay rendering, isolated so the core runs headless
  cli.py        entry point
```
