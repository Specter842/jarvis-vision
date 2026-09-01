"""
File icons: one on-screen card per real file in a folder.

`FileIcon` is the data for a single card -- where it sits, how big it is, which
real file it stands for, and (from M4 on) whether a hand is holding it.
`IconManager` scans a folder, lays the cards out in a grid, and answers
"which icon is under this point?".

M3 scope: construction, layout, and hit-testing. Nothing here moves an icon or
touches a file -- drag is M4, real file ops are M5.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import config


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

    # Held state. Set by the drag logic in M4; declared now because the build
    # spec makes "grabbed state" part of this dataclass.
    grabbed: bool = False
    grab_offset: tuple[float, float] = (0.0, 0.0)  # pinch_point - (x, y) at grab time

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
        when boxes overlap. Not wired into anything until M4.
        """
        for icon in reversed(self.icons):
            if icon.contains(point):
                return icon
        return None

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
