"""
Staged-action safety layer.

    >>> STUB <<<  Structure only. Nothing here touches the filesystem yet.
    The shapes are frozen now so Milestone 5 is a fill-in, not a rewrite.

Intended behaviour (built in M5, from the build spec):

  * A drop over a valid zone creates a PendingAction. The filesystem is NOT
    touched at drop time.
  * The PendingAction renders a yellow highlight + a countdown starting at
    config.PENDING_ACTION_SECONDS.
  * Cancelled before the countdown hits 0 (open-palm hold, or the spacebar
    dev/testing fallback) -> discard it, nothing written to disk or log.
  * Countdown reaches 0 -> commit:
        MOVE       -> shutil.move(icon.real_path, dest_folder / filename)
        QUARANTINE -> shutil.move(icon.real_path, quarantine / filename)
    QUARANTINE is a soft delete: never os.remove, never a real delete, ever,
    in v0.1.
  * On commit, append one line to config.ACTIONS_LOG_PATH:
        <ISO-8601 timestamp>\t<action_type>\t<source path>\t<dest path>
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import config

if TYPE_CHECKING:
    # icons.py arrives in M3. The annotation is a string under
    # `from __future__ import annotations`, so this import never runs at
    # runtime and the stub imports cleanly today.
    from icons import FileIcon


class ActionType(str, Enum):
    MOVE = "move"
    QUARANTINE = "quarantine"


class HudState(str, Enum):
    COUNTING_DOWN = "counting_down"
    COMMITTED = "committed"
    CANCELLED = "cancelled"


@dataclass
class PendingAction:
    """A staged filesystem action waiting out its safety countdown.

    Fields are exactly those named in the build spec:
      icon, action_type, dest_path, created_at, hud_state
    """

    icon: "FileIcon"
    action_type: ActionType
    dest_path: Path
    created_at: float                       # time.perf_counter() stamp
    hud_state: HudState = HudState.COUNTING_DOWN

    # -- M5 --------------------------------------------------------------
    def seconds_remaining(self, now: float) -> float:
        raise NotImplementedError("M5: PENDING_ACTION_SECONDS - (now - created_at)")

    def is_expired(self, now: float) -> bool:
        raise NotImplementedError("M5: seconds_remaining(now) <= 0")


class ActionManager:
    """Owns the pending-action queue and the commit / cancel / log lifecycle.

    STUB: every mutating method raises or no-ops until M5. `pending` and
    `update()` are safe to call now so main.py can wire the render + tick loop
    ahead of time.
    """

    def __init__(self) -> None:
        self._pending: list[PendingAction] = []

    @property
    def pending(self) -> list[PendingAction]:
        """Live view of staged actions, for the HUD renderer."""
        return list(self._pending)

    @property
    def has_pending(self) -> bool:
        return bool(self._pending)

    # -- M5 ------------------------------------------------------------------

    def stage(
        self,
        icon: "FileIcon",
        action_type: ActionType,
        dest_path: Path,
    ) -> PendingAction:
        """Create + enqueue a PendingAction. Does NOT touch disk."""
        raise NotImplementedError("M5")

    def cancel_all(self) -> None:
        """Discard every pending action. Nothing written to disk or log."""
        raise NotImplementedError("M5")

    def update(self, now: float) -> None:
        """Tick countdowns; commit whatever expired; drop finished entries.

        No-op until M5 so the main loop can call it unconditionally.
        """
        # M5: for action in list(self._pending):
        #         if action.is_expired(now): self._commit(action)
        #     then drop COMMITTED / CANCELLED entries.
        return None

    def _commit(self, action: PendingAction) -> None:
        """shutil.move per action_type, then append the log line."""
        # M5: MOVE / QUARANTINE both via shutil.move (never os.remove),
        #     then self._append_log(...).
        raise NotImplementedError("M5")

    def _append_log(self, action_type: ActionType, src: Path, dest: Path) -> None:
        """Append one tab-separated audit line to config.ACTIONS_LOG_PATH."""
        # M5: f"{datetime.now().isoformat()}\t{action_type.value}\t{src}\t{dest}\n"
        raise NotImplementedError("M5")
