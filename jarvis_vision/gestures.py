"""
Rule-based gesture detection. No classifiers, no ML models, no external gesture
libraries -- pixel arithmetic and per-hand state machines only.

M2 scope: the thumb-index PINCH with a hysteresis state machine, fed strictly
from post-smoothing landmarks.
M5 scope: open-palm "cancel" -- every fingertip spread far from the wrist,
held briefly.

Later milestones plug in here:
  * M6 -- two-hand resize (distance between two pinch points inside one icon)
"""

from __future__ import annotations

import time
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


class OpenPalmDetector:
    """Per-hand "all five fingers spread" detector with a hold timer.

    Spread is measured as (fingertip -> wrist) / (wrist -> middle-finger MCP),
    so it is scale-invariant. The palm must read open for HOLD_SECONDS before
    `held` latches true; `just_held` is the single frame it latches.
    """

    def __init__(self, spread_ratio: float, hold_seconds: float) -> None:
        self.spread_ratio = float(spread_ratio)
        self.hold_seconds = float(hold_seconds)

        self.open_now = False
        self.held = False
        self.just_held = False
        self._open_since: float | None = None

    def update(self, landmarks_px: np.ndarray, now: float) -> None:
        wrist = landmarks_px[config.WRIST]
        palm_ref = float(np.hypot(*(landmarks_px[config.MIDDLE_MCP] - wrist)))

        if palm_ref > 1e-3:
            self.open_now = all(
                float(np.hypot(*(landmarks_px[tip] - wrist))) / palm_ref
                >= self.spread_ratio
                for tip in config.FINGERTIPS
            )
        else:
            self.open_now = False

        was_held = self.held
        if self.open_now:
            if self._open_since is None:
                self._open_since = now
            self.held = (now - self._open_since) >= self.hold_seconds
        else:
            self._open_since = None
            self.held = False

        self.just_held = self.held and not was_held


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
    open_palm: bool = False           # 5 fingers spread, held long enough
    open_palm_just_held: bool = False # the frame open_palm latched true


class _HandSlot:
    """Everything we persist for one hand identity across frames."""

    def __init__(self) -> None:
        self.smoother = HandSmoother(config.EMA_ALPHA)
        self.pinch = PinchDetector(
            config.PINCH_ENTER_THRESHOLD_PX, config.PINCH_EXIT_THRESHOLD_PX
        )
        self.palm = OpenPalmDetector(
            config.OPEN_PALM_SPREAD_RATIO, config.OPEN_PALM_HOLD_SECONDS
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
        self,
        hands: list[Hand],
        frame_shape: tuple[int, ...],
        now: float | None = None,
    ) -> list[HandGesture]:
        if now is None:
            now = time.perf_counter()
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
            slot.palm.update(smoothed, now)

            gestures.append(
                HandGesture(
                    handedness=hand.handedness,
                    landmarks_px=smoothed,
                    pinch_state=state,
                    pinch_distance=distance,
                    pinch_point=(thumb + index) * 0.5,
                    just_pinched=slot.pinch.just_entered,
                    just_released=slot.pinch.just_released,
                    open_palm=slot.palm.held,
                    open_palm_just_held=slot.palm.just_held,
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
