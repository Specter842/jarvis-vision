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

Each file in the folder becomes a draggable icon. The folder argument is
optional — without it you get the hand-tracking / pinch demo with no icons.
Quit with `q` or `Esc`.

## Architecture

| module | responsibility |
|--------|----------------|
| `main.py` | capture loop, drawing, top-level orchestration |
| `config.py` | every tunable constant — nothing hardcoded elsewhere |
| `hand_tracker.py` | MediaPipe Tasks wrapper: frame in → list of hand landmark sets out |
| `smoothing.py` | EMA smoothing (one instance per tracked landmark, per hand) |
| `gestures.py` | pinch detection, hysteresis state machine |
| `icons.py` | `FileIcon` + `IconManager` — layout, hit-testing, grab/drag/release |
| `actions.py` | staged-action queue, pending-HUD, commit / cancel / audit log |

### Pinch, with hysteresis

`distance(thumb_tip, index_tip)` on **post-smoothing** landmarks drives a
per-hand state machine:

```
IDLE  --(dist < PINCH_ENTER_THRESHOLD_PX)-->  PINCHING
PINCHING  --(dist > PINCH_EXIT_THRESHOLD_PX)-->  IDLE
```

The gap between enter (40 px) and exit (55 px) is the anti-flicker margin — the
state never transitions on a raw distance.

### Safety layer

A drop over a valid zone does **not** touch the filesystem. It stages a
`PendingAction` with a visible countdown. If you don't cancel (open-palm hold, or
the spacebar dev fallback) before it hits zero, the action commits:

- `move` → `shutil.move(src, dest_folder / name)`
- `quarantine` → `shutil.move(src, quarantine / name)` — a soft delete, never
  `os.remove`

Every commit appends a line to `actions.log`: ISO timestamp, action type, source
path, destination path.

## Milestones

- [x] **M1** — webcam feed + 21-point hand skeleton overlay, un-mirrored
- [x] **M2** — pinch detection + hysteresis + EMA smoothing, on-screen state indicator
- [x] **M3** — file icons from a real folder (visualization only)
- [x] **M4** — drag (screen position only, no filesystem writes)
- [ ] **M5** — real file operations through the staged-action safety layer
- [ ] **M6** — two-hand resize
- [ ] **M7** — polish: 5+ files, FPS logging, threshold tuning

## Phase 2 (out of scope for v0.1)

AR / 3D / WebXR spatial interface.
