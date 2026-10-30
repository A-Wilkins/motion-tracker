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

## Running it live

```bash
python scripts/check_camera.py                 # confirm the device opens
python -m motion_tracker --flip                # mirrored, as a front-facing camera should be
python -m motion_tracker --flip --mask         # plus the motion mask, to tune the gate by eye
python -m motion_tracker --flip --no-gate      # detect every frame
```

`q` or `esc` quits, but only while the video window has focus; `ctrl-c` in the
terminal works either way and still prints the run summary.

On a 30 fps camera the pipeline is camera-bound, not compute-bound: it holds the
full 30 fps with the gate on and drops 2–4% of frames. Expect the HUD's gate
indicator to sit on ACTIVE whenever you are moving and fall to idle within about
half a second of stillness.

Two behaviours that look like bugs on a webcam pointed at yourself and are not:

**You go still, your box disappears.** MOG2 learns a background over
`motion.history` frames — ten seconds at 30fps. Sit motionless and you become
background, the gate idles, inference stops, and your track ages out and reports
`LOST`. Move again and you come back with a *new* track id, because the identity
was legitimately given up. This is correct for a driveway and wrong for a demo:
run `--no-gate` if you want stable boxes while you sit at the keyboard.

**Walking out of frame reports `EXITED`, pausing behind something reports
`LOST`.** That distinction is the point, and the clearest way to see it working
is to walk off the side of the frame, then to step behind something mid-frame.

The shipped `config.yaml` is tuned for a static scene rather than for a face at
desk distance. For a self-facing demo, raising `motion.history` keeps you in the
foreground longer and raising `track.max_age` makes identity survive longer gaps.

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

M1 MacBook, 1280x720, YOLO11n on MPS. Median of three runs, range in brackets.
Throughput excludes startup — loading weights and compiling GPU kernels is a
one-off cost and charging it to the frame rate understates steady state.

### Recorded source, 130-frame fixture — compute-bound

| | gate on | gate off |
|---|---|---|
| inferences run | 89 | 130 |
| frames skipped | 31.5% | 0% |
| throughput | **49.1 fps** [43.1–49.3] | 40.2 fps [39.7–41.3] |
| per-frame pipeline cost | 10.8 ms | 22.4 ms |
| inference, mean | 24.7 ms | 22.4 ms |

With nothing throttling input, the gate converts avoided inference directly into
throughput: ~1.22x, which is about what skipping 31.5% of an inference-bound
pipeline should buy.

### Live webcam, 30 fps — camera-bound

| | gate on | gate off |
|---|---|---|
| inferences run | 9–82% skipped, scene-dependent | 0% skipped |
| throughput | **30.2 fps** [29.9–30.4] | 27.9 fps [26.9–28.0] |
| frames dropped | 2–4% | 7–14% |
| inference, mean | 27.1 ms | 27.0–37.7 ms |

This is the more honest number and it says something different. A 30 fps camera
cannot be outrun, so the gate does not buy throughput here — it buys *headroom*.
Gated, the pipeline holds the full frame rate and discards almost nothing.
Ungated it cannot quite keep up, falls to ~28 fps, and drops 7–14% of frames to
stay current. Skip rate swings from 9% to 82% between runs purely on how much
was moving, which is the gate working as intended rather than measurement noise.

Note the one place the gate looks worse: mean inference is slightly *higher* with
it on. That is not noise. Gating means the detector only ever sees frames with
something in them, while the ungated average is diluted by empty frames that are
cheap to post-process. The win is in how many inferences are avoided, not in
making any one faster — a benchmark reporting only mean inference latency would
have concluded the gate made things worse.

Reproduce:

```bash
python -m motion_tracker --source assets/test_clip.mp4 --no-display [--no-gate]
python -m motion_tracker --source 0 --no-display --max-frames 200 [--no-gate]
```

The summary line reports `fps=<run mean> (now <rolling>)`, per-stage latency,
`wait` (time blocked on capture, which is the gap between what the pipeline
costs and what it delivers), and dropped frames for live sources.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"                 # drop [dev] to skip the test deps
python scripts/check_camera.py          # verify camera access
python -m motion_tracker
```

On macOS, camera access is granted to the **application that owns your shell**,
not to the shell itself. In an editor's integrated terminal that is the editor,
so the entry to look for under System Settings → Privacy & Security → Camera is
VS Code or Cursor, not "python" or "zsh".

An app only appears in that list once it has requested access, and it can only
request access if its `Info.plist` declares `NSCameraUsageDescription`.
**Terminal.app does not declare it**, so a camera request from Terminal.app is
denied without ever prompting and Terminal will never show up there. Use an
editor's integrated terminal or iTerm2 instead.

The prompt only appears for a foreground, interactive run, so run
`check_camera.py` yourself rather than through any sandboxed or automated
wrapper. If access was denied once before, clear the stale decision first with
`tccutil reset Camera <bundle-id>` — `check_camera.py` prints the exact command
for your setup.

## Tests

```bash
pytest              # 41 unit tests, no model weights needed
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
