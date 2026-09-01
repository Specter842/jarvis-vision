"""
File icons: one on-screen card per real file in a folder.

`FileIcon` is the data for a single card -- where it sits, how big it is, which
real file it stands for, and whether a hand is holding it.
`IconManager` scans a folder, lays the cards out in a grid, answers "which icon
is under this point?", and runs the grab / drag / release cycle each frame.

M4 scope: drag by pinch -- screen position only. Nothing here touches a file;
drop zones and real file ops are M5, two-hand resize is M6.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

import config
from gestures import HandGesture, PinchState


@dataclass
class FileIcon:
    """One draggable card standing in for a real file on disk."""

    real_path: Path
    label: str                       # display name (usually real_path.name)

    # Top-left corner in displayed-frame pixel space, and current size.
    # Size is per-icon (not the config constant) so M6 can resize one icon
    # without touching the rest.
    x: float
    y: float
    w: float
    h: float

    # Held state.
    grabbed: bool = False
    grab_offset: tuple[float, float] = (0.0, 0.0)  # pinch_point - (x, y) at grab time
    # Which hand ("Left"/"Right") holds this icon. Needed so a second hand's
    # IDLE state cannot release an icon the first hand is dragging, and so two
    # hands can drag two different icons at once.
    grabbed_by: str | None = None

    # Where the icon sat before the current grab, so a cancelled action (M5)
    # can put it back exactly.
    home: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if self.home is None:
            self.home = (self.x, self.y)

    # -- geometry -------------------------------------------------------------

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """(left, top, right, bottom) in pixels."""
        return (self.x, self.y, self.x + self.w, self.y + self.h)

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)

    def contains(self, point) -> bool:
        """Is `point` (x, y) inside this icon's bounding box?"""
        px, py = float(point[0]), float(point[1])
        left, top, right, bottom = self.bbox
        return left <= px <= right and top <= py <= bottom


class IconManager:
    """Owns the set of icons for one folder."""

    def __init__(self, icons: list[FileIcon], source_folder: Path) -> None:
        self.icons = icons
        self.source_folder = source_folder
        self.frame_shape: tuple[int, ...] = (0, 0, 0)  # set by layout_grid

    # -- construction -------------------------------------------------------

    @classmethod
    def from_folder(
        cls, folder: Path, frame_shape: tuple[int, ...]
    ) -> "IconManager":
        """Build one icon per regular file directly inside `folder`.

        Non-recursive, files only (sub-folders become drop targets in M5, not
        icons). Sorted by name for a stable, repeatable layout.
        """
        folder = Path(folder).expanduser().resolve()
        if not folder.is_dir():
            raise NotADirectoryError(f"{folder} is not a folder")

        entries = sorted(
            (p for p in folder.iterdir() if p.is_file()),
            key=lambda p: p.name.lower(),
        )

        iw, ih = config.ICON_SIZE_PX
        icons = [
            FileIcon(
                real_path=p,
                label=_fit_label(p.name, config.ICON_LABEL_MAX_CHARS),
                x=0.0, y=0.0, w=float(iw), h=float(ih),
            )
            for p in entries
        ]

        manager = cls(icons, folder)
        manager.layout_grid(frame_shape)
        return manager

    # -- layout ----------------------------------------------------------------

    def layout_grid(self, frame_shape: tuple[int, ...]) -> None:
        """Place icons left-to-right, top-to-bottom in a grid that fits the frame width."""
        self.frame_shape = frame_shape
        frame_h, frame_w = frame_shape[0], frame_shape[1]

        origin_x, origin_y = config.ICON_GRID_ORIGIN_PX
        gap = config.ICON_GRID_SPACING_PX
        label_gap = config.ICON_LABEL_GAP_PX
        iw, ih = config.ICON_SIZE_PX

        cell_w = iw + gap
        cell_h = ih + gap + label_gap

        usable_w = max(cell_w, frame_w - origin_x - gap)
        columns = max(1, int(usable_w // cell_w))

        for i, icon in enumerate(self.icons):
            row, col = divmod(i, columns)
            icon.x = float(origin_x + col * cell_w)
            icon.y = float(origin_y + row * cell_h)
            icon.w = float(iw)
            icon.h = float(ih)
            icon.home = (icon.x, icon.y)

    # -- queries -------------------------------------------------------------

    def hit_test(self, point) -> FileIcon | None:
        """Topmost icon whose box contains `point`, or None.

        Iterates back-to-front so the last-drawn (visually on top) icon wins
        when boxes overlap. Grabbed icons are moved to the end of the list, so
        they win ties both here and on screen.
        """
        for icon in reversed(self.icons):
            if icon.contains(point):
                return icon
        return None

    # -- grab / drag / release --------------------------------------------

    def apply_gestures(self, gestures: list[HandGesture]) -> None:
        """Run one frame of the grab -> drag -> release cycle.

        Rules (from the build spec):
          GRAB    -- on IDLE->PINCHING, hit-test the pinch point; if it lands
                     on a free icon, mark it grabbed and store the offset from
                     the icon origin to the pinch point.
          DRAG    -- while grabbed, icon origin = pinch point - stored offset.
          RELEASE -- on PINCHING->IDLE, drop the icon where it is. (M5 will
                     check drop zones here and maybe stage a PendingAction.)

        No filesystem access. Positions are transient -- they reset on restart.
        """
        by_hand = {g.handedness: g for g in gestures}

        # GRAB: a fresh pinch that lands on a free icon.
        for gesture in gestures:
            if not gesture.just_pinched:
                continue
            icon = self.hit_test(gesture.pinch_point)
            if icon is None or icon.grabbed:
                continue
            px, py = float(gesture.pinch_point[0]), float(gesture.pinch_point[1])
            icon.grabbed = True
            icon.grabbed_by = gesture.handedness
            icon.grab_offset = (px - icon.x, py - icon.y)
            icon.home = (icon.x, icon.y)
            self._raise_to_top(icon)

        # DRAG / RELEASE for everything currently held.
        for icon in self.icons:
            if not icon.grabbed:
                continue

            gesture = by_hand.get(icon.grabbed_by or "")
            if gesture is None:
                # The holding hand left the frame -- drop the icon in place.
                self._release(icon)
                continue

            if gesture.just_released or gesture.pinch_state is PinchState.IDLE:
                # M5: inspect drop zones with icon at this final position.
                self._release(icon)
                continue

            px, py = float(gesture.pinch_point[0]), float(gesture.pinch_point[1])
            ox, oy = icon.grab_offset
            icon.x, icon.y = self._clamp(px - ox, py - oy, icon)

    def _release(self, icon: FileIcon) -> None:
        icon.grabbed = False
        icon.grabbed_by = None

    def _raise_to_top(self, icon: FileIcon) -> None:
        """Move `icon` to the end of the draw/hit-test order."""
        self.icons.remove(icon)
        self.icons.append(icon)

    def _clamp(self, x: float, y: float, icon: FileIcon) -> tuple[float, float]:
        """Keep a dragged icon fully inside the frame."""
        frame_h, frame_w = self.frame_shape[0], self.frame_shape[1]
        if frame_w <= 0 or frame_h <= 0:
            return x, y
        x = float(np.clip(x, 0.0, max(0.0, frame_w - icon.w)))
        y = float(np.clip(y, 0.0, max(0.0, frame_h - icon.h)))
        return x, y

    def __iter__(self):
        return iter(self.icons)

    def __len__(self) -> int:
        return len(self.icons)


def _fit_label(name: str, max_chars: int) -> str:
    """Truncate an over-long filename, keeping the extension visible.

        "a_very_long_module_name.py"  ->  "a_very_long_m….py"
    """
    if len(name) <= max_chars:
        return name

    stem, dot, ext = name.rpartition(".")
    if dot and len(ext) + 2 < max_chars:
        keep = max_chars - len(ext) - 2  # room for "…" + "." + ext
        return f"{stem[:keep]}….{ext}"
    return name[: max_chars - 1] + "…"
