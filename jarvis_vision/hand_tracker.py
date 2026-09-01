"""
MediaPipe wrapper: frame in -> list of hand landmark sets out.

Uses the MediaPipe **Tasks** API (mediapipe.tasks.python.vision.HandLandmarker)
in VIDEO running mode. The legacy `mediapipe.solutions.hands` API is
deliberately NOT used anywhere in this project.

This module knows nothing about gestures, icons, or files. It converts frames
into geometry and stops there.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

import config


@dataclass
class Hand:
    """One detected hand, already converted to pixel space."""

    # "Left" or "Right", from the viewer's perspective of the displayed frame.
    handedness: str
    handedness_score: float

    # (21, 2) float array of x/y pixel coordinates in the DISPLAYED frame.
    landmarks_px: np.ndarray

    # (21, 3) float array of raw normalized MediaPipe coords (x, y in 0..1, z
    # relative depth). Kept for anything that needs depth or resolution-
    # independent math later.
    landmarks_norm: np.ndarray = field(repr=False)

    def point(self, index: int) -> np.ndarray:
        """Pixel-space (x, y) of a single landmark index."""
        return self.landmarks_px[index]


class HandTracker:
    """Thin, stateful wrapper around a VIDEO-mode HandLandmarker."""

    def __init__(self) -> None:
        model_path = config.HAND_LANDMARKER_MODEL_PATH
        if not model_path.exists():
            raise FileNotFoundError(
                f"Hand landmark model not found at {model_path}.\n"
                "Download it from the MediaPipe model zoo:\n"
                "  https://storage.googleapis.com/mediapipe-models/"
                "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            )

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.VIDEO,
            num_hands=config.MAX_NUM_HANDS,
            min_hand_detection_confidence=config.MIN_HAND_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=config.MIN_HAND_PRESENCE_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
        )
        self._landmarker = HandLandmarker.create_from_options(options)

        # VIDEO mode demands strictly increasing timestamps. We clamp rather
        # than trust the caller, because a repeated or backwards timestamp
        # makes the task throw and kill the capture loop.
        self._last_timestamp_ms = -1

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # -- inference ---------------------------------------------------------

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int) -> list[Hand]:
        """
        Run detection on one frame.

        `frame_bgr` must already be in its FINAL displayed orientation (i.e.
        mirroring is the caller's job and must happen before this call).
        Handedness labels are only meaningful under that contract -- see the
        MIRROR_FRAME note in config.py.
        """
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms

        height, width = frame_bgr.shape[:2]

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=frame_rgb)

        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        hands: list[Hand] = []
        for i, landmark_set in enumerate(result.hand_landmarks):
            norm = np.array(
                [(lm.x, lm.y, lm.z) for lm in landmark_set], dtype=np.float32
            )

            # Normalized -> pixel space. MediaPipe can return values slightly
            # outside 0..1 when a hand is partially out of frame; we keep them
            # unclamped so drawing degrades gracefully instead of snapping
            # landmarks onto the frame border.
            px = np.column_stack((norm[:, 0] * width, norm[:, 1] * height))

            label, score = self._handedness_for(result, i)
            hands.append(
                Hand(
                    handedness=label,
                    handedness_score=score,
                    landmarks_px=px,
                    landmarks_norm=norm,
                )
            )

        return hands

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _handedness_for(result, index: int) -> tuple[str, float]:
        """Extract the handedness label/score for hand `index`."""
        try:
            category = result.handedness[index][0]
            label = category.category_name
            score = float(category.score)
        except (IndexError, AttributeError):
            return "Unknown", 0.0

        if config.SWAP_HANDEDNESS:
            label = {"Left": "Right", "Right": "Left"}.get(label, label)

        return label, score
