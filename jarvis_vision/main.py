"""
JARVIS-Vision v0.1 -- Milestone 3: file icons from a real folder (view only).

Top-level orchestration only: open the camera, pump frames through the hand
tracker and the gesture engine, draw the result, show it. No file operations
and no interaction with the icons yet -- actions.py and the icon hit-testing
are wired but inert this milestone.

Run:
    .venv\\Scripts\\python jarvis_vision\\main.py PATH\\TO\\FOLDER

The folder argument is optional; without it you get the M2 pinch demo with no
icons.

Quit with `q` or ESC.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import config
from actions import ActionManager
from gestures import GestureEngine, HandGesture, PinchState
from icons import FileIcon, IconManager


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def _text(frame: np.ndarray, label: str, org: tuple[int, int], color, scale: float) -> None:
    """Draw text with a 1px shadow so it stays readable over a busy feed."""
    x, y = org
    cv2.putText(
        frame, label, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, scale,
        config.COLOR_TEXT_SHADOW, config.FONT_THICKNESS + 1, cv2.LINE_AA,
    )
    cv2.putText(
        frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
        color, config.FONT_THICKNESS, cv2.LINE_AA,
    )


def draw_skeleton(frame: np.ndarray, landmarks_px: np.ndarray, handedness: str) -> None:
    """Draw the 21-point skeleton (smoothed landmarks) and a handedness label."""
    points = landmarks_px.astype(int)

    for start, end in config.HAND_CONNECTIONS:
        cv2.line(
            frame, tuple(points[start]), tuple(points[end]),
            config.COLOR_CONNECTION, config.CONNECTION_THICKNESS_PX, cv2.LINE_AA,
        )

    for point in points:
        cv2.circle(
            frame, tuple(point), config.LANDMARK_RADIUS_PX,
            config.COLOR_LANDMARK, -1, cv2.LINE_AA,
        )

    color = (
        config.COLOR_RIGHT_HAND if handedness == "Right"
        else config.COLOR_LEFT_HAND
    )
    wrist_x, wrist_y = points[config.WRIST]
    _text(frame, handedness, (wrist_x - 24, wrist_y + 28), color, config.FONT_SCALE)


def draw_pinch(frame: np.ndarray, gesture: HandGesture) -> None:
    """Draw the pinch indicator: thumb-index line + pinch-point circle.

    Colour follows the STATE MACHINE, not the raw distance -- white while IDLE,
    green while PINCHING, and it stays green through the whole 40..55px
    hysteresis band without flicker.
    """
    pinching = gesture.pinch_state is PinchState.PINCHING
    color = config.COLOR_PINCH_ACTIVE if pinching else config.COLOR_PINCH_IDLE
    line_thickness = (
        config.PINCH_LINE_THICKNESS_ACTIVE if pinching
        else config.PINCH_LINE_THICKNESS_IDLE
    )

    landmarks = gesture.landmarks_px.astype(int)
    thumb = tuple(landmarks[config.THUMB_TIP])
    index = tuple(landmarks[config.INDEX_TIP])
    center = tuple(gesture.pinch_point.astype(int))

    cv2.line(frame, thumb, index, color, line_thickness, cv2.LINE_AA)
    cv2.circle(frame, center, config.PINCH_POINT_RADIUS_PX, color, 2, cv2.LINE_AA)
    if pinching:
        cv2.circle(frame, center, config.PINCH_POINT_RADIUS_PX - 5, color, -1, cv2.LINE_AA)

    cx, cy = center
    _text(
        frame,
        f"{gesture.pinch_state.value}  {gesture.pinch_distance:4.0f}px",
        (cx + config.PINCH_POINT_RADIUS_PX + 6, cy + 4),
        color,
        config.HUD_FONT_SCALE,
    )


def draw_icon(frame: np.ndarray, icon: FileIcon) -> None:
    """Draw one file card: translucent body, border, folded corner, name below."""
    left, top, right, bottom = (int(round(v)) for v in icon.bbox)
    corner = config.ICON_CORNER_PX

    # Translucent fill: blend a solid rectangle with the underlying feed so the
    # camera image still reads through the icon.
    roi = frame[max(top, 0):max(bottom, 0), max(left, 0):max(right, 0)]
    if roi.size:
        fill = np.full_like(roi, config.COLOR_ICON_FILL, dtype=np.uint8)
        cv2.addWeighted(fill, config.ICON_FILL_ALPHA, roi, 1 - config.ICON_FILL_ALPHA, 0, roi)

    border_color = (
        config.COLOR_ICON_GRABBED_BORDER if icon.grabbed
        else config.COLOR_ICON_BORDER
    )
    cv2.rectangle(frame, (left, top), (right, bottom), border_color,
                  config.ICON_BORDER_THICKNESS_PX, cv2.LINE_AA)

    # Folded top-right corner.
    cv2.line(frame, (right - corner, top), (right, top + corner),
             config.COLOR_ICON_CORNER, config.ICON_BORDER_THICKNESS_PX, cv2.LINE_AA)

    # Filename, centred under the card.
    (text_w, _), _ = cv2.getTextSize(
        icon.label, cv2.FONT_HERSHEY_SIMPLEX, config.HUD_FONT_SCALE, config.FONT_THICKNESS
    )
    text_x = int(icon.center[0] - text_w / 2)
    text_y = bottom + 16
    _text(frame, icon.label, (text_x, text_y), config.COLOR_ICON_LABEL, config.HUD_FONT_SCALE)


def draw_icons(frame: np.ndarray, icon_manager: IconManager | None) -> None:
    if icon_manager is None:
        return
    for icon in icon_manager:
        draw_icon(frame, icon)


def draw_hud(
    frame: np.ndarray,
    fps: float,
    gestures: list[HandGesture],
    icon_manager: IconManager | None,
) -> None:
    """Top-left status readout + milestone banner."""
    _text(frame, f"FPS {fps:5.1f}", (12, 26), config.COLOR_TEXT, config.HUD_FONT_SCALE)
    _text(
        frame,
        f"enter <{config.PINCH_ENTER_THRESHOLD_PX:.0f}  exit >{config.PINCH_EXIT_THRESHOLD_PX:.0f}"
        f"  alpha {config.EMA_ALPHA}",
        (12, 48), config.COLOR_TEXT, config.HUD_FONT_SCALE,
    )
    if icon_manager is not None:
        _text(
            frame,
            f"{len(icon_manager)} icons  <- {icon_manager.source_folder}",
            (12, frame.shape[0] - 36), config.COLOR_TEXT, config.HUD_FONT_SCALE,
        )

    y = 70
    for gesture in gestures:
        pinching = gesture.pinch_state is PinchState.PINCHING
        color = config.COLOR_PINCH_ACTIVE if pinching else config.COLOR_TEXT
        _text(
            frame,
            f"{gesture.handedness:<5} {gesture.pinch_state.value:<9} {gesture.pinch_distance:4.0f}px",
            (12, y), color, config.HUD_FONT_SCALE,
        )
        y += 22

    _text(
        frame, "M3: file icons (view only)  |  q / ESC to quit",
        (12, frame.shape[0] - 14), config.COLOR_TEXT, config.HUD_FONT_SCALE,
    )


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------


def open_camera() -> cv2.VideoCapture:
    """
    Open the configured camera.

    CAP_DSHOW is requested first: on Windows the default MSMF backend can take
    several seconds to hand over the first frame, and on some drivers never
    does. We fall back to the default backend if DirectShow refuses.
    """
    capture = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_DSHOW)
    if not capture.isOpened():
        capture.release()
        capture = cv2.VideoCapture(config.CAMERA_INDEX)

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open camera index {config.CAMERA_INDEX}. "
            "Close any other app using the webcam, or change CAMERA_INDEX in config.py."
        )

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAPTURE_WIDTH)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAPTURE_HEIGHT)
    return capture


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="jarvis-vision",
        description="Gesture-controlled file interface (v0.1, on-screen only).",
    )
    parser.add_argument(
        "folder",
        nargs="?",
        type=Path,
        help="folder whose files are shown as draggable icons (M3+). "
        "Omit for the pinch demo with no icons.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    folder: Path | None = None
    if args.folder is not None:
        folder = args.folder.expanduser()
        if not folder.is_dir():
            print(f"[jarvis-vision] not a folder: {folder}", file=sys.stderr)
            return 2

    capture = open_camera()
    actual_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[jarvis-vision] camera {config.CAMERA_INDEX} open at {actual_w}x{actual_h}")
    print(f"[jarvis-vision] mirror={config.MIRROR_FRAME} swap_handedness={config.SWAP_HANDEDNESS}")

    engine = GestureEngine()
    actions = ActionManager()  # instantiated to prove the wiring; no-op until M5
    icon_manager: IconManager | None = None  # built on the first frame, once size is known

    fps = 0.0
    last_time = time.perf_counter()
    start_time = last_time

    try:
        # HandTracker imports mediapipe, which is slow -- do it inside main so
        # --help and import errors surface fast.
        from hand_tracker import HandTracker

        with HandTracker() as tracker:
            print("[jarvis-vision] model loaded, entering capture loop")

            while True:
                ok, frame = capture.read()
                if not ok:
                    print("[jarvis-vision] dropped frame from camera, stopping", file=sys.stderr)
                    break

                # Mirror BEFORE inference. Handedness is only correct when the
                # frame we detect on is the frame we display.
                if config.MIRROR_FRAME:
                    frame = cv2.flip(frame, 1)

                # Lay out icons once we know the true displayed frame size
                # (the driver may ignore the requested capture resolution).
                if folder is not None and icon_manager is None:
                    icon_manager = IconManager.from_folder(folder, frame.shape)
                    print(f"[jarvis-vision] loaded {len(icon_manager)} icons from {folder}")

                timestamp_ms = int((time.perf_counter() - start_time) * 1000)
                hands = tracker.process(frame, timestamp_ms)
                gestures = engine.update(hands, frame.shape)

                # Icons are the "desktop": draw them first, hands/cursor on top.
                draw_icons(frame, icon_manager)

                for gesture in gestures:
                    draw_skeleton(frame, gesture.landmarks_px, gesture.handedness)
                    draw_pinch(frame, gesture)

                actions.update(time.perf_counter())  # no-op stub until M5

                now = time.perf_counter()
                delta = now - last_time
                last_time = now
                if delta > 0:
                    instant = 1.0 / delta
                    fps = (
                        instant if fps == 0.0
                        else (config.FPS_SMOOTHING_ALPHA * instant
                              + (1 - config.FPS_SMOOTHING_ALPHA) * fps)
                    )

                draw_hud(frame, fps, gestures, icon_manager)
                cv2.imshow(config.WINDOW_NAME, frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):  # q or ESC
                    break

                # Closing the window with the X button should also exit.
                if cv2.getWindowProperty(config.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
    finally:
        capture.release()
        cv2.destroyAllWindows()

    print("[jarvis-vision] shut down cleanly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
