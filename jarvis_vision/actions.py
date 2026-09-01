"""
Staged-action safety layer.

A drop over a valid zone does NOT touch the filesystem. It stages a
`PendingAction` with a visible countdown (config.PENDING_ACTION_SECONDS). If it
is not cancelled before the countdown hits zero, it commits:

    MOVE       -> shutil.move(icon.real_path, dest_dir / filename)
    QUARANTINE -> shutil.move(icon.real_path, QUARANTINE_DIR / filename)

QUARANTINE is a soft delete -- never os.remove, never a real delete, ever, in
v0.1. Every commit appends one tab-separated line to actions.log:

    <ISO-8601 timestamp>\t<action_type>\t<source path>\t<dest path>

Cancel (open-palm hold, or the spacebar dev fallback) discards every pending
action, snaps its icon back to where it was picked up, and writes nothing.
"""

from __future__ import annotations

import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import config

if TYPE_CHECKING:
    from icons import FileIcon


class ActionType(str, Enum):
    MOVE = "move"
    QUARANTINE = "quarantine"


class HudState(str, Enum):
    COUNTING_DOWN = "counting_down"
    COMMITTED = "committed"
    CANCELLED = "cancelled"
    FAILED = "failed"          # commit raised (e.g. source vanished); nothing logged


@dataclass
class PendingAction:
    """A staged filesystem action waiting out its safety countdown.

    Fields are exactly those named in the build spec:
      icon, action_type, dest_path, created_at, hud_state
    """

    icon: "FileIcon"
    action_type: ActionType
    dest_path: Path                         # full final path, including filename
    created_at: float                       # time.perf_counter() stamp
    hud_state: HudState = HudState.COUNTING_DOWN

    def seconds_remaining(self, now: float) -> float:
        return max(0.0, config.PENDING_ACTION_SECONDS - (now - self.created_at))

    def is_expired(self, now: float) -> bool:
        return (now - self.created_at) >= config.PENDING_ACTION_SECONDS


class ActionManager:
    """Owns the pending-action queue and the commit / cancel / log lifecycle."""

    def __init__(self) -> None:
        self._pending: list[PendingAction] = []

    @property
    def pending(self) -> list[PendingAction]:
        """Live view of staged actions, for the HUD renderer."""
        return list(self._pending)

    @property
    def has_pending(self) -> bool:
        return bool(self._pending)

    # -- staging -----------------------------------------------------------

    def stage(
        self,
        icon: "FileIcon",
        action_type: str | ActionType,
        dest_dir: Path,
    ) -> PendingAction:
        """Create + enqueue a PendingAction. Does NOT touch disk."""
        at = ActionType(action_type)
        dest_path = _unique_destination(Path(dest_dir), icon.real_path.name)
        action = PendingAction(
            icon=icon,
            action_type=at,
            dest_path=dest_path,
            created_at=time.perf_counter(),
        )
        icon.pending = True
        self._pending.append(action)
        return action

    # -- cancel ----------------------------------------------------------------

    def cancel_all(self) -> int:
        """Discard every pending action; snap each icon home. Returns the count."""
        if not self._pending:
            return 0
        n = len(self._pending)
        for action in self._pending:
            action.hud_state = HudState.CANCELLED
            action.icon.pending = False
            if action.icon.home is not None:
                action.icon.x, action.icon.y = action.icon.home
        self._pending.clear()
        return n

    # -- tick ----------------------------------------------------------------

    def update(self, now: float) -> None:
        """Commit whatever expired this frame; keep the rest counting down."""
        survivors: list[PendingAction] = []
        for action in self._pending:
            if action.is_expired(now):
                self._commit(action)
            else:
                survivors.append(action)
        self._pending = survivors

    # -- commit ----------------------------------------------------------------

    def _commit(self, action: PendingAction) -> None:
        src = Path(action.icon.real_path)
        dest = action.dest_path
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            # shutil.move raises shutil.Error (not OSError) on a dest-name
            # collision or a move-into-self; catch both so a failed commit
            # surfaces as the FAILED HUD state instead of killing the loop.
            shutil.move(str(src), str(dest))
        except (OSError, shutil.Error) as exc:
            action.hud_state = HudState.FAILED
            # If the source is already gone the move partially succeeded (or
            # something else took it): drop the icon rather than leaving a
            # draggable card pointing at a path that no longer exists.
            if not src.exists():
                action.icon.pending = False
                action.icon.committed = True
            else:
                action.icon.pending = False
            print(f"[jarvis-vision] action failed: {src} -> {dest}: {exc}", file=sys.stderr)
            return

        action.hud_state = HudState.COMMITTED
        action.icon.pending = False
        action.icon.committed = True
        self._append_log(action.action_type, src, dest)

    def _append_log(self, action_type: ActionType, src: Path, dest: Path) -> None:
        line = "\t".join(
            (
                datetime.now().isoformat(timespec="seconds"),
                action_type.value,
                str(src),
                str(dest),
            )
        ) + "\n"
        with open(config.ACTIONS_LOG_PATH, "a", encoding="utf-8") as handle:
            handle.write(line)


def _unique_destination(dest_dir: Path, name: str) -> Path:
    """A path inside `dest_dir` that does not already exist -- never clobber."""
    candidate = dest_dir / name
    if not candidate.exists():
        return candidate
    stem, suffix = candidate.stem, candidate.suffix
    i = 1
    while (dest_dir / f"{stem} ({i}){suffix}").exists():
        i += 1
    return dest_dir / f"{stem} ({i}){suffix}"
