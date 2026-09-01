"""
EMA (exponential moving average) smoothing.

Detector output jitters frame to frame. A raw thumb-index distance will cross
any fixed threshold back and forth several times a second just from noise, so
every value that feeds a gesture decision is smoothed here first -- the state
machines in gestures.py never see a raw coordinate.

    smoothed += alpha * (measurement - smoothed)

Higher alpha -> follows the measurement faster, smooths less. See EMA_ALPHA in
config.py.
"""

from __future__ import annotations

import numpy as np


class EMASmoother:
    """Exponential moving average of a single scalar or fixed-shape vector.

    One instance holds the running estimate for one tracked value. The first
    measurement seeds the estimate directly (no ramp-in lag); every subsequent
    call eases the estimate toward the new measurement by `alpha`.
    """

    def __init__(self, alpha: float) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.alpha = float(alpha)
        self._value: np.ndarray | None = None

    def update(self, measurement) -> np.ndarray:
        """Fold in a new measurement and return the updated estimate.

        The returned array is the smoother's live buffer -- copy it before
        storing a long-lived reference.
        """
        m = np.asarray(measurement, dtype=np.float32)
        if self._value is None or self._value.shape != m.shape:
            self._value = m.copy()
        else:
            self._value += self.alpha * (m - self._value)
        return self._value

    @property
    def value(self) -> np.ndarray | None:
        return self._value

    @property
    def initialized(self) -> bool:
        return self._value is not None

    def reset(self) -> None:
        """Forget history. The next measurement seeds a fresh estimate."""
        self._value = None


class HandSmoother:
    """One EMASmoother per landmark, for a single tracked hand.

    Smoothing each of the 21 landmarks independently (rather than the array as
    one blob) keeps the option open of per-landmark alphas later without a
    structural change.
    """

    def __init__(self, alpha: float, num_landmarks: int = 21) -> None:
        self._smoothers = [EMASmoother(alpha) for _ in range(num_landmarks)]

    def update(self, landmarks_px: np.ndarray) -> np.ndarray:
        """(N, 2) raw pixel landmarks in -> (N, 2) smoothed pixel landmarks out.

        Returns a fresh array each call; safe to keep.
        """
        out = np.empty((len(self._smoothers), 2), dtype=np.float32)
        for i, smoother in enumerate(self._smoothers):
            out[i] = smoother.update(landmarks_px[i])
        return out

    def reset(self) -> None:
        for smoother in self._smoothers:
            smoother.reset()
