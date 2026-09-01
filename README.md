# JARVIS-Vision

A gesture-controlled file interface. Point a webcam at your hand, pinch to grab a
file icon on screen, drag it onto a folder or the trash zone, and — after a
safety countdown you can cancel — the real file moves on disk.

v0.1 is **on-screen only** (a `cv2` window). No AR, no 3D, no WebXR.

Built on:

| piece | role |
|-------|------|
| [OpenCV](https://pypi.org/project/opencv-contrib-python/) (`cv2`) | camera capture, drawing, window |
| [MediaPipe](https://ai.google.dev/edge/mediapipe) **Tasks API** | `HandLandmarker` in `VIDEO` mode — 21-point hand tracking |
| NumPy | landmark arithmetic |
| `shutil` / `pathlib` | the real file operations |

Gesture recognition is **rule-based arithmetic only** — pixel distances and
per-hand state machines. No classifiers, no training, no gesture libraries.

## Setup

```powershell
py -3.14 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python scripts\download_model.py
```

The hand-landmark model (`hand_landmarker.task`, ~7.6 MB) is a build artifact and
is not committed. `download_model.py` pulls it from the official MediaPipe model
zoo; the exact URL is also recorded in [`jarvis_vision/config.py`](jarvis_vision/config.py).

## Run

```powershell
.venv\Scripts\python jarvis_vision\main.py "C:\path\to\a\folder"
```

Each file in the folder becomes a draggable icon; each sub-folder becomes a
"move here" drop zone, alongside a fixed trash zone. Pinch an icon onto a zone
and a countdown starts — let it finish to move/quarantine the real file, or
cancel with an open-palm hold (or `Space`). Pinch the **same icon with both
hands** and move your hands apart / together to resize it. The folder argument
is optional — without it you get the hand-tracking / pinch demo with no icons.
Quit with `q` or `Esc`.

## Architecture

| module | responsibility |
|--------|----------------|
| `main.py` | capture loop, drawing, top-level orchestration |
| `config.py` | every tunable constant — nothing hardcoded elsewhere |
| `hand_tracker.py` | MediaPipe Tasks wrapper: frame in → list of hand landmark sets out |
| `smoothing.py` | EMA smoothing (one instance per tracked landmark, per hand) |
| `gestures.py` | pinch detection + hysteresis, open-palm cancel, two-hand-pinch span |
| `icons.py` | `FileIcon` + `DropZone` + `IconManager` — layout, hit-testing, grab/drag/release, two-hand resize |
| `actions.py` | staged-action queue, pending-HUD, commit / cancel / audit log |

### Pinch, with hysteresis

`distance(thumb_tip, index_tip)` on **post-smoothing** landmarks drives a
per-hand state machine:

```
IDLE  --(dist < PINCH_ENTER_THRESHOLD_PX)-->  PINCHING
PINCHING  --(dist > PINCH_EXIT_THRESHOLD_PX)-->  IDLE
```

The gap between enter (40 px) and exit (55 px) is the anti-flicker margin — the
state never transitions on a raw distance. See [Tuning](#tuning) for why these
numbers.

### Safety layer

A drop over a valid zone does **not** touch the filesystem. It stages a
`PendingAction` with a visible countdown. If you don't cancel (open-palm hold, or
the spacebar dev fallback) before it hits zero, the action commits:

- `move` → `shutil.move(src, dest_folder / name)`
- `quarantine` → `shutil.move(src, quarantine / name)` — a soft delete, never
  `os.remove`

Every commit appends a line to `actions.log`: ISO timestamp, action type, source
path, destination path.

## Tuning

Every value below lives in [`config.py`](jarvis_vision/config.py) and is changeable
without touching another file. These are the **final chosen values** for v0.1 and
why.

### `EMA_ALPHA = 0.5`  *(raised from 0.4 in M7)*

`smoothed += alpha * (measurement - smoothed)`, applied per landmark, per hand.

- The **hysteresis gap does the anti-flicker work**, not the EMA — so the filter
  doesn't need to be heavy. It only has to kill single-frame landmark jitter.
- At 30 fps, `0.5` reaches 90 % of a step in ~3.3 frames (~110 ms): drag tracks
  the hand closely, with no rubber-banding you can see.
- `0.4` was ~150 ms — a touch laggy on quick drags. Above ~`0.6` raw jitter
  starts to show: the pinch point visibly buzzes when the hand is held still.

### Pinch: `ENTER = 40 px`, `EXIT = 55 px`  *(15 px gap)*

Thumb-tip ↔ index-tip distance, in pixels, on the **smoothed** landmarks.

- **40 to enter**: at a comfortable arm's length from a 720p webcam the tips are
  ~35–45 px apart when they touch (finger width + model bias mean the distance
  never reaches 0). 40 catches a deliberate pinch without demanding a hard press.
- **55 to exit**: landmark noise on the tips is ~3–6 px RMS at 720p; a 15 px gap
  is ~3× that, so noise alone can never walk `PINCHING → IDLE`. Releasing takes a
  deliberate ~1.5 cm finger separation.
- Asymmetry is the point: easy to grab, hard to drop by accident.

### Open-palm cancel: `SPREAD_RATIO = 1.5`, `HOLD = 0.3 s`

Ratio = `(fingertip → wrist) / (wrist → middle-finger MCP)`, required for **all
five** tips. Scale-invariant, so it works as the hand moves toward/away from the
camera.

- Open hand: tips sit at 1.6–3.0× the palm length. Fist: 1.0–1.4×. `1.5` is the
  empty band between, with the **thumb** (smallest open ratio, ~1.6–1.8×) as the
  limiting finger.
- `0.3 s` hold stops a passing flat-hand pose — e.g. the moment you release a
  pinch — from cancelling. You have to mean it.

### Resize: `MIN/MAX = 40 / 240 px`, `MIN_BASELINE = 24 px`

- Icon scales by `current span / entry span` between the two pinch points,
  clamped to 40–240 px.
- The two points can start anywhere from ~0 to ~110 px apart inside an 80 px
  icon. Below ~24 px the ratio's denominator is small enough that hand tremor
  causes large scale swings, so a resize won't start from there.

### FPS

Measured FPS (EMA, `FPS_SMOOTHING_ALPHA = 0.1`) is logged every
`FPS_LOG_INTERVAL_SECONDS` (5 s), with a `WARNING` to stderr whenever it's under
`TARGET_FPS_WARN_THRESHOLD` (20). A per-session `avg / min / max` summary prints
on exit. If the average is under target, drop `CAPTURE_WIDTH/HEIGHT`.

### Layout

The icon grid uses the widest column count the frame allows, then scales
**icons + spacing together** to fit the height between the HUD and the drop-zone
band. Any file count lays out with no overlap, nothing off-screen, and nothing
drawn over the drop zones — icons just get smaller as the folder grows.

## Milestones

- [x] **M1** — webcam feed + 21-point hand skeleton overlay, un-mirrored
- [x] **M2** — pinch detection + hysteresis + EMA smoothing, on-screen state indicator
- [x] **M3** — file icons from a real folder (visualization only)
- [x] **M4** — drag (screen position only, no filesystem writes)
- [x] **M5** — real file operations through the staged-action safety layer
- [x] **M6** — two-hand resize
- [x] **M7** — polish: grid scales to any file count, FPS logged + summarised, gesture constants tuned & documented above

**v0.1 complete.**

## Phase 2 (out of scope for v0.1)

AR / 3D / WebXR spatial interface.
