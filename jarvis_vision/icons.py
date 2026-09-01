"""
File icons and drop zones: one on-screen card per real file in a folder, plus
the targets you can drop them on.

`FileIcon` is the data for a single card -- where it sits, how big it is, which
real file it stands for, and whether a hand is holding it / an action is staged.
`DropZone` is a folder ("move here") or the trash ("quarantine") target.
`IconManager` scans a folder, lays cards and zones out, answers "which icon /
zone is under this point?", and runs the grab / drag / release cycle each frame.

M5 scope: on release over a valid zone, report the drop so the caller can stage
a `PendingAction`. This module still never touches the filesystem -- `actions.py`
does that, only after the safety countdown.
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

    # Safety-layer state (M5). `pending` = a PendingAction is counting down for
    # this icon (locked, cannot be re-grabbed). `committed` = the file has been
    # moved; IconManager drops the icon next frame.
    pending: bool = False
    committed: bool = False

    # Where the icon sat before the current grab, so a cancelled action
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

    @property
    def locked(self) -> bool:
        """Cannot be grabbed -- a countdown is running or the file is gone."""
        return self.pending or self.committed


@dataclass
class DropZone:
    """A place a file icon can be dropped to stage a real action."""

    kind: str            # "folder" | "trash"
    action_type: str     # "move" | "quarantine"  (matches actions.ActionType)
    dest_dir: Path       # directory the file is moved into on commit
    label: str

    x: float
    y: float
    w: float
    h: float

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (self.x, self.y, self.x + self.w, self.y + self.h)

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)

    def contains(self, point) -> bool:
        px, py = float(point[0]), float(point[1])
        left, top, right, bottom = self.bbox
        return left <= px <= right and top <= py <= bottom


class IconManager:
    """Owns the icons and drop zones for one folder."""

    def __init__(
        self,
        icons: list[FileIcon],
        drop_zones: list[DropZone],
        source_folder: Path,
    ) -> None:
        self.icons = icons
        self.drop_zones = drop_zones
        self.source_folder = source_folder
        self.frame_shape: tuple[int, ...] = (0, 0, 0)  # set by layout_grid

    # -- construction -------------------------------------------------------

    @classmethod
    def from_folder(
        cls, folder: Path, frame_shape: tuple[int, ...]
    ) -> "IconManager":
        """Build one icon per regular file, one drop zone per sub-folder, plus
        the trash zone.

        Non-recursive. Files and sub-folders are sorted by name for a stable,
        repeatable layout.
        """
        folder = Path(folder).expanduser().resolve()
        if not folder.is_dir():
            raise NotADirectoryError(f"{folder} is not a folder")

        files = sorted(
            (p for p in folder.iterdir() if p.is_file()),
            key=lambda p: p.name.lower(),
        )
        subdirs = sorted(
            (p for p in folder.iterdir() if p.is_dir()),
            key=lambda p: p.name.lower(),
        )

        iw, ih = config.ICON_SIZE_PX
        icons = [
            FileIcon(
                real_path=p,
                label=_fit_label(p.name, config.ICON_LABEL_MAX_CHARS),
                x=0.0, y=0.0, w=float(iw), h=float(ih),
            )
            for p in files
        ]

        zw, zh = config.DROP_ZONE_SIZE_PX
        drop_zones = [
            DropZone(
                kind="folder", action_type="move", dest_dir=d,
                label=_fit_label(d.name, config.ICON_LABEL_MAX_CHARS),
                x=0.0, y=0.0, w=float(zw), h=float(zh),
            )
            for d in subdirs[: config.MAX_FOLDER_DROP_ZONES]
        ]
        drop_zones.append(
            DropZone(
                kind="trash", action_type="quarantine",
                dest_dir=config.QUARANTINE_DIR, label="QUARANTINE",
                x=0.0, y=0.0, w=float(zw), h=float(zh),
            )
        )

        manager = cls(icons, drop_zones, folder)
        manager.layout_grid(frame_shape)
        return manager

    # -- layout ----------------------------------------------------------------

    def layout_grid(self, frame_shape: tuple[int, ...]) -> None:
        """Place icons in a top-left grid and drop zones in a bottom band."""
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

        self._layout_drop_zones(frame_shape)

    def _layout_drop_zones(self, frame_shape: tuple[int, ...]) -> None:
        """A single row along the bottom: folders from the left, trash last."""
        frame_h, frame_w = frame_shape[0], frame_shape[1]
        zw, zh = config.DROP_ZONE_SIZE_PX
        gap = config.DROP_ZONE_GAP_PX
        side = config.DROP_ZONE_SIDE_MARGIN_PX
        y = float(frame_h - config.DROP_ZONE_BOTTOM_MARGIN_PX - zh)

        x = float(side)
        for zone in self.drop_zones:
            zone.w, zone.h, zone.y = float(zw), float(zh), y
            if zone.kind == "trash":
                zone.x = float(max(x, frame_w - side - zw))
            else:
                zone.x = x
                x += zw + gap

    # -- queries -------------------------------------------------------------

    def hit_test(self, point) -> FileIcon | None:
        """Topmost grab-able icon whose box contains `point`, or None.

        Iterates back-to-front so the last-drawn (visually on top) icon wins
        when boxes overlap. Grabbed icons are moved to the end of the list, so
        they win ties both here and on screen.
        """
        for icon in reversed(self.icons):
            if not icon.locked and icon.contains(point):
                return icon
        return None

    def drop_zone_at(self, point) -> DropZone | None:
        """The drop zone whose box contains `point`, or None."""
        for zone in self.drop_zones:
            if zone.contains(point):
                return zone
        return None

    # -- grab / drag / release --------------------------------------------

    def apply_gestures(
        self, gestures: list[HandGesture]
    ) -> list[tuple[FileIcon, DropZone]]:
        """Run one frame of the grab -> drag -> release cycle.

        Rules (from the build spec):
          GRAB    -- on IDLE->PINCHING, hit-test the pinch point; if it lands
                     on a free icon, mark it grabbed and store the offset from
                     the icon origin to the pinch point.
          DRAG    -- while grabbed, icon origin = pinch point - stored offset.
          RELEASE -- on PINCHING->IDLE: if the icon's centre is over a drop
                     zone, hand it back for staging (icon stays put, locked);
                     otherwise drop it where it is.

        Returns the (icon, zone) pairs released over a zone this frame. Still
        no filesystem access -- the caller stages a PendingAction.
        """
        by_hand = {g.handedness: g for g in gestures}
        drops: list[tuple[FileIcon, DropZone]] = []

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
                zone = self.drop_zone_at(icon.center)
                icon.grabbed = False
                icon.grabbed_by = None
                if zone is not None:
                    icon.pending = True  # locked until commit or cancel
                    drops.append((icon, zone))
                continue

            px, py = float(gesture.pinch_point[0]), float(gesture.pinch_point[1])
            ox, oy = icon.grab_offset
            icon.x, icon.y = self._clamp(px - ox, py - oy, icon)

        return drops

    def remove_committed(self) -> list[FileIcon]:
        """Drop icons whose file has been moved. Returns the removed icons."""
        gone = [ic for ic in self.icons if ic.committed]
        if gone:
            self.icons = [ic for ic in self.icons if not ic.committed]
        return gone

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
