"""
Rule-based gesture detection. No classifiers, no ML models, no external gesture
libraries -- pixel arithmetic and per-hand state machines only.

M2 scope: the thumb-index PINCH with a hysteresis state machine, fed strictly
from post-smoothing landmarks.

Later milestones plug in here:
  * M4 -- grab/drag consumes `just_pinched` / `just_released` + `pinch_point`
  * M5 -- open-palm "cancel" detection (all 5 fingertip->wrist spreads high)
  * M6 -- two-hand resize (distance between two pinch points inside one icon)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

import config
from hand_tracker import Hand
from smoothing import HandSmoother


class PinchState(Enum):
    IDLE = "IDLE"
    PINCHING = "PINCHING"


class PinchDetector:
    """Per-hand hysteresis state machine for distance(thumb_tip, index_tip).

        IDLE  --(dist < ENTER)-->  PINCHING  --(dist > EXIT)-->  IDLE

    The gap between ENTER and EXIT is the anti-flicker margin: once pinching,
    the fingers must open well past the trigger distance before the pinch
    releases. `update()` acts only on the distance it is given, which the
    caller MUST compute from smoothed landmarks.
    """

    def __init__(self, enter_px: float, exit_px: float) -> None:
        if exit_px <= enter_px:
            raise ValueError(
                f"exit threshold ({exit_px}) must exceed enter threshold "
                f"({enter_px}) -- that gap is the hysteresis"
            )
        self.enter_px = float(enter_px)
        self.exit_px = float(exit_px)

        self.state = PinchState.IDLE
        self.distance = float("inf")
        # One-frame edge flags, cleared at the top of every update().
        self.just_entered = False
        self.just_released = False

    def update(self, distance_px: float) -> PinchState:
        self.distance = float(distance_px)
        self.just_entered = False
        self.just_released = False

        if self.state is PinchState.IDLE:
            if self.distance < self.enter_px:
                self.state = PinchState.PINCHING
                self.just_entered = True
        else:  # PINCHING
            if self.distance > self.exit_px:
                self.state = PinchState.IDLE
                self.just_released = True

        return self.state

    def reset(self) -> None:
        self.state = PinchState.IDLE
        self.distance = float("inf")
        self.just_entered = False
        self.just_released = False


@dataclass
class HandGesture:
    """This frame's gesture readout for one hand."""

    handedness: str
    landmarks_px: np.ndarray          # smoothed (21, 2), pixel space
    pinch_state: PinchState
    pinch_distance: float             # smoothed thumb-index distance, px
    pinch_point: np.ndarray           # (2,) smoothed thumb/index midpoint
    just_pinched: bool                # IDLE -> PINCHING this frame
    just_released: bool               # PINCHING -> IDLE this frame


class _HandSlot:
    """Everything we persist for one hand identity across frames."""

    def __init__(self) -> None:
        self.smoother = HandSmoother(config.EMA_ALPHA)
        self.pinch = PinchDetector(
            config.PINCH_ENTER_THRESHOLD_PX, config.PINCH_EXIT_THRESHOLD_PX
        )


class GestureEngine:
    """Owns all cross-frame gesture state.

    main.py passes this frame's raw detections; we return this frame's gesture
    readouts. Hand identity is keyed by handedness label, which is unique while
    MAX_NUM_HANDS == 2; an unlabelled hand falls back to a left/right-of-centre
    key so it still gets a stable smoother.
    """

    def __init__(self) -> None:
        self._slots: dict[str, _HandSlot] = {}

    def update(
        self, hands: list[Hand], frame_shape: tuple[int, ...]
    ) -> list[HandGesture]:
        frame_width = frame_shape[1]
        seen: set[str] = set()
        gestures: list[HandGesture] = []

        for hand in hands:
            key = self._key_for(hand, frame_width)
            if key in seen:
                # Detector briefly reported two hands with the same identity.
                # Give the second one its own transient slot rather than
                # double-updating the first.
                key = f"{key}#dup"
            seen.add(key)

            slot = self._slots.get(key)
            if slot is None:
                slot = _HandSlot()
                self._slots[key] = slot

            smoothed = slot.smoother.update(hand.landmarks_px)
            thumb = smoothed[config.THUMB_TIP]
            index = smoothed[config.INDEX_TIP]
            distance = float(np.hypot(thumb[0] - index[0], thumb[1] - index[1]))
            state = slot.pinch.update(distance)

            gestures.append(
                HandGesture(
                    handedness=hand.handedness,
                    landmarks_px=smoothed,
                    pinch_state=state,
                    pinch_distance=distance,
                    pinch_point=(thumb + index) * 0.5,
                    just_pinched=slot.pinch.just_entered,
                    just_released=slot.pinch.just_released,
                )
            )

        # Forget hands that left the frame so a returning hand seeds a fresh
        # estimate instead of easing in from a stale position.
        for stale in [k for k in self._slots if k not in seen]:
            del self._slots[stale]

        return gestures

    @staticmethod
    def _key_for(hand: Hand, frame_width: int) -> str:
        if hand.handedness in ("Left", "Right"):
            return hand.handedness
        wrist_x = float(hand.landmarks_px[config.WRIST][0])
        return "hand@L" if wrist_x < frame_width / 2 else "hand@R"


# ---------------------------------------------------------------------------
# M6 -- two-hand resize detection will live here (not built this milestone).
# ---------------------------------------------------------------------------
