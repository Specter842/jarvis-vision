"""
JARVIS-Vision v0.1 -- Milestone 1: webcam feed + hand skeleton overlay.

Top-level orchestration only: open the camera, pump frames through the hand
tracker, draw the result, show it. No gesture logic, no file operations.

Run:
    .venv\\Scripts\\python jarvis_vision\\main.py

Quit with `q` or ESC.
"""

from __future__ import annotations

import sys
import time

import cv2
import numpy as np

import config
from hand_tracker import Hand, HandTracker


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


def draw_hand(frame: np.ndarray, hand: Hand) -> None:
    """Draw the 21-point skeleton and a handedness label for one hand."""
    points = hand.landmarks_px.astype(int)

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
        config.COLOR_RIGHT_HAND if hand.handedness == "Right"
        else config.COLOR_LEFT_HAND
    )
    wrist_x, wrist_y = points[config.WRIST]
    _text(
        frame,
        f"{hand.handedness} {hand.handedness_score:.2f}",
        (wrist_x - 30, wrist_y + 28),
        color,
        config.FONT_SCALE,
    )


def draw_hud(frame: np.ndarray, fps: float, hand_count: int) -> None:
    """Top-left status readout."""
    _text(frame, f"FPS {fps:5.1f}", (12, 26), config.COLOR_TEXT, config.HUD_FONT_SCALE)
    _text(frame, f"Hands {hand_count}", (12, 48), config.COLOR_TEXT, config.HUD_FONT_SCALE)
    _text(
        frame, "M1: tracking only  |  q / ESC to quit",
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
# Main loop
# ---------------------------------------------------------------------------


def main() -> int:
    capture = open_camera()
    actual_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[jarvis-vision] camera {config.CAMERA_INDEX} open at {actual_w}x{actual_h}")
    print(f"[jarvis-vision] mirror={config.MIRROR_FRAME} swap_handedness={config.SWAP_HANDEDNESS}")

    fps = 0.0
    last_time = time.perf_counter()
    start_time = last_time

    try:
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

                timestamp_ms = int((time.perf_counter() - start_time) * 1000)
                hands = tracker.process(frame, timestamp_ms)

                for hand in hands:
                    draw_hand(frame, hand)

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

                draw_hud(frame, fps, len(hands))
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
    raise SystemExit(main())
