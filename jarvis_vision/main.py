"""
JARVIS-Vision v0.1 -- Milestone 5: real file operations via the safety layer.

Top-level orchestration only: open the camera, pump frames through the hand
tracker and the gesture engine, run the grab/drag/release cycle, tick the
staged-action queue, draw everything, show it.

Drop a file icon on a folder zone or the trash zone and a yellow countdown
starts. Let it reach zero and the real file moves (folder) or is quarantined
(trash) -- logged to actions.log. Cancel before zero with an open-palm hold or
the SPACE key and nothing happens.

Run:
    .venv\\Scripts\\python jarvis_vision\\main.py PATH\\TO\\FOLDER

The folder argument is optional; without it you get the pinch demo with no
icons. Sub-folders of the given folder become "move" drop zones.

Quit with `q` or ESC.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import config
from actions import ActionManager, HudState, PendingAction
from gestures import GestureEngine, HandGesture, PinchState
from icons import FileIcon, IconManager


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


def draw_skeleton(frame: np.ndarray, landmarks_px: np.ndarray, handedness: str) -> None:
    """Draw the 21-point skeleton (smoothed landmarks) and a handedness label."""
    points = landmarks_px.astype(int)

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
        config.COLOR_RIGHT_HAND if handedness == "Right"
        else config.COLOR_LEFT_HAND
    )
    wrist_x, wrist_y = points[config.WRIST]
    _text(frame, handedness, (wrist_x - 24, wrist_y + 28), color, config.FONT_SCALE)


def draw_pinch(frame: np.ndarray, gesture: HandGesture) -> None:
    """Draw the pinch indicator: thumb-index line + pinch-point circle.

    Colour follows the STATE MACHINE, not the raw distance -- white while IDLE,
    green while PINCHING, and it stays green through the whole 40..55px
    hysteresis band without flicker.
    """
    pinching = gesture.pinch_state is PinchState.PINCHING
    color = config.COLOR_PINCH_ACTIVE if pinching else config.COLOR_PINCH_IDLE
    line_thickness = (
        config.PINCH_LINE_THICKNESS_ACTIVE if pinching
        else config.PINCH_LINE_THICKNESS_IDLE
    )

    landmarks = gesture.landmarks_px.astype(int)
    thumb = tuple(landmarks[config.THUMB_TIP])
    index = tuple(landmarks[config.INDEX_TIP])
    center = tuple(gesture.pinch_point.astype(int))

    cv2.line(frame, thumb, index, color, line_thickness, cv2.LINE_AA)
    cv2.circle(frame, center, config.PINCH_POINT_RADIUS_PX, color, 2, cv2.LINE_AA)
    if pinching:
        cv2.circle(frame, center, config.PINCH_POINT_RADIUS_PX - 5, color, -1, cv2.LINE_AA)

    cx, cy = center
    _text(
        frame,
        f"{gesture.pinch_state.value}  {gesture.pinch_distance:4.0f}px",
        (cx + config.PINCH_POINT_RADIUS_PX + 6, cy + 4),
        color,
        config.HUD_FONT_SCALE,
    )


def draw_icon(frame: np.ndarray, icon: FileIcon) -> None:
    """Draw one file card: translucent body, border, folded corner, name below."""
    left, top, right, bottom = (int(round(v)) for v in icon.bbox)
    corner = config.ICON_CORNER_PX

    # Translucent fill: blend a solid rectangle with the underlying feed so the
    # camera image still reads through the icon.
    roi = frame[max(top, 0):max(bottom, 0), max(left, 0):max(right, 0)]
    if roi.size:
        fill = np.full_like(roi, config.COLOR_ICON_FILL, dtype=np.uint8)
        cv2.addWeighted(fill, config.ICON_FILL_ALPHA, roi, 1 - config.ICON_FILL_ALPHA, 0, roi)

    if icon.resizing:
        border_color = config.COLOR_ICON_RESIZING_BORDER
    elif icon.grabbed:
        border_color = config.COLOR_ICON_GRABBED_BORDER
    else:
        border_color = config.COLOR_ICON_BORDER
    cv2.rectangle(frame, (left, top), (right, bottom), border_color,
                  config.ICON_BORDER_THICKNESS_PX, cv2.LINE_AA)

    # Folded top-right corner.
    cv2.line(frame, (right - corner, top), (right, top + corner),
             config.COLOR_ICON_CORNER, config.ICON_BORDER_THICKNESS_PX, cv2.LINE_AA)

    if icon.resizing:
        _text(frame, f"{int(round(icon.w))}px", (left, top - 8),
              config.COLOR_ICON_RESIZING_BORDER, config.HUD_FONT_SCALE)

    # Filename, centred under the card.
    (text_w, _), _ = cv2.getTextSize(
        icon.label, cv2.FONT_HERSHEY_SIMPLEX, config.HUD_FONT_SCALE, config.FONT_THICKNESS
    )
    text_x = int(icon.center[0] - text_w / 2)
    text_y = bottom + 16
    _text(frame, icon.label, (text_x, text_y), config.COLOR_ICON_LABEL, config.HUD_FONT_SCALE)


def draw_icons(frame: np.ndarray, icon_manager: IconManager | None) -> None:
    if icon_manager is None:
        return
    for icon in icon_manager:
        draw_icon(frame, icon)


def _translucent_rect(frame: np.ndarray, bbox, color, alpha: float) -> None:
    left, top, right, bottom = (int(round(v)) for v in bbox)
    roi = frame[max(top, 0):max(bottom, 0), max(left, 0):max(right, 0)]
    if roi.size:
        fill = np.full_like(roi, color, dtype=np.uint8)
        cv2.addWeighted(fill, alpha, roi, 1 - alpha, 0, roi)


def draw_drop_zones(
    frame: np.ndarray,
    icon_manager: IconManager | None,
    gestures: list[HandGesture],
) -> None:
    """Folder zones (blue) + trash zone (red); cyan while a grabbed icon hovers."""
    if icon_manager is None:
        return

    hovered = {
        id(icon_manager.drop_zone_at(ic.center))
        for ic in icon_manager
        if ic.grabbed
    }

    for zone in icon_manager.drop_zones:
        base = (
            config.COLOR_DROP_TRASH if zone.kind == "trash"
            else config.COLOR_DROP_FOLDER
        )
        hot = id(zone) in hovered
        color = config.COLOR_DROP_ZONE_HOT if hot else base

        _translucent_rect(frame, zone.bbox, color, config.DROP_ZONE_FILL_ALPHA)
        left, top, right, bottom = (int(round(v)) for v in zone.bbox)
        cv2.rectangle(frame, (left, top), (right, bottom), color,
                      config.DROP_ZONE_BORDER_PX + (1 if hot else 0), cv2.LINE_AA)

        verb = "quarantine" if zone.kind == "trash" else "move here"
        (tw, _), _ = cv2.getTextSize(
            zone.label, cv2.FONT_HERSHEY_SIMPLEX, config.HUD_FONT_SCALE, config.FONT_THICKNESS
        )
        _text(frame, zone.label, (int(zone.center[0] - tw / 2), top + 26),
              config.COLOR_DROP_ZONE_LABEL, config.HUD_FONT_SCALE)
        (vw, _), _ = cv2.getTextSize(
            verb, cv2.FONT_HERSHEY_SIMPLEX, config.HUD_FONT_SCALE * 0.8, config.FONT_THICKNESS
        )
        _text(frame, verb, (int(zone.center[0] - vw / 2), bottom - 12),
              config.COLOR_DROP_ZONE_LABEL, config.HUD_FONT_SCALE * 0.8)


def draw_pending(frame: np.ndarray, actions: list[PendingAction], now: float) -> None:
    """Yellow highlight box + numeric countdown on top of each staged icon."""
    for action in actions:
        if action.hud_state is not HudState.COUNTING_DOWN:
            continue
        icon = action.icon
        pad = config.PENDING_BOX_INFLATE_PX
        left, top, right, bottom = icon.bbox
        cv2.rectangle(
            frame,
            (int(left - pad), int(top - pad)),
            (int(right + pad), int(bottom + pad)),
            config.COLOR_PENDING_BOX, config.PENDING_BOX_THICKNESS_PX, cv2.LINE_AA,
        )

        secs = f"{action.seconds_remaining(now):.1f}"
        (tw, th), _ = cv2.getTextSize(
            secs, cv2.FONT_HERSHEY_SIMPLEX, config.PENDING_COUNTDOWN_FONT_SCALE, 2
        )
        cx, cy = icon.center
        _text(frame, secs, (int(cx - tw / 2), int(cy + th / 2)),
              config.COLOR_PENDING_TEXT, config.PENDING_COUNTDOWN_FONT_SCALE)

        caption = f"{action.action_type.value} -> {action.dest_path.parent.name}"
        (cw, _), _ = cv2.getTextSize(
            caption, cv2.FONT_HERSHEY_SIMPLEX, config.HUD_FONT_SCALE, config.FONT_THICKNESS
        )
        _text(frame, caption, (int(cx - cw / 2), int(top - pad - 8)),
              config.COLOR_PENDING_TEXT, config.HUD_FONT_SCALE)


def draw_hud(
    frame: np.ndarray,
    fps: float,
    gestures: list[HandGesture],
    icon_manager: IconManager | None,
    actions: ActionManager,
) -> None:
    """Top-left status readout + milestone banner."""
    _text(frame, f"FPS {fps:5.1f}", (12, 26), config.COLOR_TEXT, config.HUD_FONT_SCALE)
    _text(
        frame,
        f"enter <{config.PINCH_ENTER_THRESHOLD_PX:.0f}  exit >{config.PINCH_EXIT_THRESHOLD_PX:.0f}"
        f"  alpha {config.EMA_ALPHA}",
        (12, 48), config.COLOR_TEXT, config.HUD_FONT_SCALE,
    )
    if icon_manager is not None:
        held = [ic for ic in icon_manager if ic.grabbed]
        summary = f"{len(icon_manager)} icons  <- {icon_manager.source_folder}"
        if held:
            summary += "   holding: " + ", ".join(
                f"{ic.label} ({ic.grabbed_by})" for ic in held
            )
        _text(
            frame, summary,
            (12, frame.shape[0] - 36),
            config.COLOR_PINCH_ACTIVE if held else config.COLOR_TEXT,
            config.HUD_FONT_SCALE,
        )

    y = 70
    for gesture in gestures:
        pinching = gesture.pinch_state is PinchState.PINCHING
        color = config.COLOR_PINCH_ACTIVE if pinching else config.COLOR_TEXT
        tag = "  OPEN-PALM" if gesture.open_palm else ""
        _text(
            frame,
            f"{gesture.handedness:<5} {gesture.pinch_state.value:<9} {gesture.pinch_distance:4.0f}px{tag}",
            (12, y),
            config.COLOR_PENDING_TEXT if gesture.open_palm else color,
            config.HUD_FONT_SCALE,
        )
        y += 22

    if actions.has_pending:
        _text(
            frame,
            f"{len(actions.pending)} action(s) pending -- OPEN PALM or SPACE to cancel",
            (12, y + 4), config.COLOR_PENDING_TEXT, config.HUD_FONT_SCALE,
        )
        y += 22

    if icon_manager is not None and icon_manager.resizing_icon is not None:
        ic = icon_manager.resizing_icon
        _text(
            frame, f"RESIZING {ic.label}  {int(round(ic.w))}px "
            f"[{config.RESIZE_MIN_PX}-{config.RESIZE_MAX_PX}]",
            (12, y + 4), config.COLOR_ICON_RESIZING_BORDER, config.HUD_FONT_SCALE,
        )

    _text(
        frame, "M6: two-hand resize  |  open-palm / SPACE cancels  |  q / ESC quit",
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
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="jarvis-vision",
        description="Gesture-controlled file interface (v0.1, on-screen only).",
    )
    parser.add_argument(
        "folder",
        nargs="?",
        type=Path,
        help="folder whose files are shown as draggable icons (M3+). "
        "Omit for the pinch demo with no icons.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    folder: Path | None = None
    if args.folder is not None:
        folder = args.folder.expanduser()
        if not folder.is_dir():
            print(f"[jarvis-vision] not a folder: {folder}", file=sys.stderr)
            return 2

    capture = open_camera()
    actual_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[jarvis-vision] camera {config.CAMERA_INDEX} open at {actual_w}x{actual_h}")
    print(f"[jarvis-vision] mirror={config.MIRROR_FRAME} swap_handedness={config.SWAP_HANDEDNESS}")
    print(f"[jarvis-vision] tuning: EMA_ALPHA={config.EMA_ALPHA} "
          f"pinch enter/exit={config.PINCH_ENTER_THRESHOLD_PX}/{config.PINCH_EXIT_THRESHOLD_PX}px "
          f"pending={config.PENDING_ACTION_SECONDS}s")

    engine = GestureEngine()
    actions = ActionManager()
    icon_manager: IconManager | None = None  # built on the first frame, once size is known

    fps = 0.0
    last_time = time.perf_counter()
    start_time = last_time

    # FPS accounting for the periodic log + shutdown summary.
    frame_count = 0
    fps_min = float("inf")
    fps_max = 0.0
    last_fps_log = last_time

    try:
        # HandTracker imports mediapipe, which is slow -- do it inside main so
        # --help and import errors surface fast.
        from hand_tracker import HandTracker

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

                # Lay out icons once we know the true displayed frame size
                # (the driver may ignore the requested capture resolution).
                if folder is not None and icon_manager is None:
                    icon_manager = IconManager.from_folder(folder, frame.shape)
                    print(f"[jarvis-vision] loaded {len(icon_manager)} icons from {folder}")

                now = time.perf_counter()
                timestamp_ms = int((now - start_time) * 1000)
                hands = tracker.process(frame, timestamp_ms)
                gestures = engine.update(hands, frame.shape, now)

                # Grab / drag / release. A release over a drop zone comes back
                # as (icon, zone) -- stage it, but do NOT touch the filesystem.
                if icon_manager is not None:
                    for icon, zone in icon_manager.apply_gestures(gestures):
                        action = actions.stage(icon, zone.action_type, zone.dest_dir)
                        print(f"[jarvis-vision] staged {action.action_type.value}: "
                              f"{icon.real_path.name} -> {action.dest_path}")

                # Cancel triggers: open-palm hold (rising edge) on any hand.
                if any(g.open_palm_just_held for g in gestures):
                    cancelled = actions.cancel_all()
                    if cancelled:
                        print(f"[jarvis-vision] open-palm cancel: {cancelled} action(s) discarded")

                # Tick the safety countdown; commit whatever expired.
                actions.update(now)
                if icon_manager is not None:
                    for removed in icon_manager.remove_committed():
                        print(f"[jarvis-vision] committed: {removed.real_path.name}")

                # -- render -------------------------------------------------
                draw_drop_zones(frame, icon_manager, gestures)
                draw_icons(frame, icon_manager)          # the "desktop"
                draw_pending(frame, actions.pending, now)

                # Two-hand resize: a bar between the two pinch points.
                if icon_manager is not None and icon_manager.resizing_icon is not None:
                    pts = [g.pinch_point for g in gestures
                           if g.pinch_state is PinchState.PINCHING]
                    if len(pts) == 2:
                        cv2.line(frame, tuple(pts[0].astype(int)), tuple(pts[1].astype(int)),
                                 config.COLOR_ICON_RESIZING_BORDER, 2, cv2.LINE_AA)

                for gesture in gestures:
                    draw_skeleton(frame, gesture.landmarks_px, gesture.handedness)
                    draw_pinch(frame, gesture)

                delta = now - last_time
                last_time = now
                frame_count += 1
                if delta > 0:
                    instant = 1.0 / delta
                    fps = (
                        instant if fps == 0.0
                        else (config.FPS_SMOOTHING_ALPHA * instant
                              + (1 - config.FPS_SMOOTHING_ALPHA) * fps)
                    )
                    # Ignore the first ~15 frames -- model warm-up skews them.
                    if frame_count > 15:
                        fps_min = min(fps_min, fps)
                        fps_max = max(fps_max, fps)

                # Periodic FPS log + sub-threshold warning (rate-limited).
                if now - last_fps_log >= config.FPS_LOG_INTERVAL_SECONDS:
                    last_fps_log = now
                    if fps < config.TARGET_FPS_WARN_THRESHOLD and frame_count > 15:
                        print(f"[jarvis-vision] WARNING: fps {fps:.1f} below "
                              f"target {config.TARGET_FPS_WARN_THRESHOLD}", file=sys.stderr)
                    else:
                        print(f"[jarvis-vision] fps {fps:.1f} ({frame_count} frames)")

                draw_hud(frame, fps, gestures, icon_manager, actions)
                cv2.imshow(config.WINDOW_NAME, frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):  # q or ESC
                    break
                if key == 32:  # SPACE -- dev/testing cancel fallback
                    cancelled = actions.cancel_all()
                    if cancelled:
                        print(f"[jarvis-vision] SPACE cancel: {cancelled} action(s) discarded")

                # Closing the window with the X button should also exit.
                if cv2.getWindowProperty(config.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
    finally:
        capture.release()
        cv2.destroyAllWindows()

    elapsed = time.perf_counter() - start_time
    avg = frame_count / elapsed if elapsed > 0 else 0.0
    lo = fps_min if fps_min != float("inf") else 0.0
    print(f"[jarvis-vision] session: {frame_count} frames in {elapsed:.1f}s  "
          f"fps avg {avg:.1f}  min {lo:.1f}  max {fps_max:.1f}")
    if avg and avg < config.TARGET_FPS_WARN_THRESHOLD:
        print(f"[jarvis-vision] NOTE: average fps was below the "
              f"{config.TARGET_FPS_WARN_THRESHOLD} target -- try a lower capture "
              f"resolution (CAPTURE_WIDTH/HEIGHT in config.py).", file=sys.stderr)
    print("[jarvis-vision] shut down cleanly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
