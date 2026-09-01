"""
Central configuration for JARVIS-Vision v0.1.

EVERY tunable constant lives here. No magic numbers anywhere else in the
codebase -- if you find yourself wanting to hardcode a value in another
module, add it here instead.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

# MediaPipe hand landmark model (Tasks API bundle).
#
# Downloaded from the official MediaPipe model zoo. Exact URL used:
#   https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
#
# This is the float16 variant, revision 1 -- the build the MediaPipe docs
# point at for the HandLandmarker task. ~7.6 MB.
HAND_LANDMARKER_MODEL_PATH = PROJECT_ROOT / "hand_landmarker.task"

QUARANTINE_DIR = PROJECT_ROOT / "quarantine"  # soft-delete target (M5)
ACTIONS_LOG_PATH = PROJECT_ROOT / "actions.log"  # append-only audit log (M5)


# ---------------------------------------------------------------------------
# Camera / window
# ---------------------------------------------------------------------------

CAMERA_INDEX = 0
WINDOW_NAME = "JARVIS-Vision v0.1"

# Requested capture resolution. The driver may ignore these and hand back a
# different size; all downstream math uses the ACTUAL frame size, never these.
CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720

# Mirror the frame horizontally so the feed behaves like a mirror: move your
# hand right, the on-screen hand moves right. This is what makes direct
# manipulation feel correct in later milestones.
#
# It also matters for handedness. MediaPipe determines handedness ASSUMING the
# input image is mirrored (selfie-camera convention). Because we mirror before
# inference, the label it returns is already correct and passes through
# untouched. See SWAP_HANDEDNESS below.
MIRROR_FRAME = True

# Escape hatch: if your camera driver already mirrors in hardware (some laptop
# webcams and virtual cams do), handedness will come out inverted. Flip this to
# True to swap the labels without touching any other file.
SWAP_HANDEDNESS = False


# ---------------------------------------------------------------------------
# Hand tracking
# ---------------------------------------------------------------------------

MAX_NUM_HANDS = 2

MIN_HAND_DETECTION_CONFIDENCE = 0.5
MIN_HAND_PRESENCE_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5


# ---------------------------------------------------------------------------
# Gesture thresholds  (M2+)
# ---------------------------------------------------------------------------

PINCH_ENTER_THRESHOLD_PX = 40  # thumb-tip/index-tip distance to START a pinch
PINCH_EXIT_THRESHOLD_PX = 55   # distance to END a pinch -- the hysteresis gap

EMA_ALPHA = 0.4  # smoothing: higher = more responsive, less smooth


# ---------------------------------------------------------------------------
# Staged actions / safety layer  (M5)
# ---------------------------------------------------------------------------

PENDING_ACTION_SECONDS = 1.5  # HUD countdown before a drop commits


# ---------------------------------------------------------------------------
# Icons  (M3+)
# ---------------------------------------------------------------------------

ICON_SIZE_PX = (80, 80)

RESIZE_MIN_PX = 40   # two-hand resize clamp, lower bound (M6)
RESIZE_MAX_PX = 240  # two-hand resize clamp, upper bound (M6)

# Grid layout for icons loaded from a folder (M3). Positions are in displayed-
# frame pixel space and recomputed whenever the frame size is known.
ICON_GRID_ORIGIN_PX = (40, 130)   # top-left of the first icon; clears the HUD
ICON_GRID_SPACING_PX = 28         # gap between icon cells, both axes
ICON_LABEL_GAP_PX = 20            # vertical room reserved under each icon for its name
ICON_LABEL_MAX_CHARS = 16         # longer names get middle-truncated with an ellipsis


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------

TARGET_FPS_WARN_THRESHOLD = 20  # warn if measured fps drops below this

FPS_SMOOTHING_ALPHA = 0.1  # EMA over the on-screen fps readout so it is legible


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

# BGR, because OpenCV.
COLOR_LANDMARK = (0, 255, 255)     # yellow -- joint dots
COLOR_CONNECTION = (0, 200, 0)     # green  -- bones
COLOR_TEXT = (255, 255, 255)       # white
COLOR_TEXT_SHADOW = (0, 0, 0)      # black
COLOR_LEFT_HAND = (255, 160, 0)    # blue-ish
COLOR_RIGHT_HAND = (0, 160, 255)   # orange-ish

LANDMARK_RADIUS_PX = 4
CONNECTION_THICKNESS_PX = 2

FONT_SCALE = 0.6
FONT_THICKNESS = 1
HUD_FONT_SCALE = 0.55

# Pinch indicator (M2). The circle sits on the pinch point (thumb/index
# midpoint) and switches colour on the state-machine transition, NOT on the
# raw distance -- that is what the hysteresis gap buys us.
COLOR_PINCH_IDLE = (255, 255, 255)      # white -- IDLE
COLOR_PINCH_ACTIVE = (0, 255, 0)        # green -- PINCHING
PINCH_POINT_RADIUS_PX = 14
PINCH_LINE_THICKNESS_IDLE = 2
PINCH_LINE_THICKNESS_ACTIVE = 4

# File icons (M3). Drawn as a filled card with a border and a folded corner,
# name centred underneath.
COLOR_ICON_FILL = (48, 42, 36)          # dark slate
COLOR_ICON_BORDER = (210, 210, 210)     # light grey
COLOR_ICON_CORNER = (150, 150, 150)     # folded-corner accent
COLOR_ICON_LABEL = (255, 255, 255)      # white
COLOR_ICON_GRABBED_BORDER = (0, 255, 0) # green while grabbed (unused until M4)
ICON_BORDER_THICKNESS_PX = 2
ICON_CORNER_PX = 14                     # size of the folded corner triangle
ICON_FILL_ALPHA = 0.55                  # icon card translucency over the feed


# ---------------------------------------------------------------------------
# Hand topology
# ---------------------------------------------------------------------------

# Named landmark indices for the 21-point MediaPipe hand model.
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20

FINGERTIPS = (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)

# The standard 21-bone hand skeleton, declared explicitly so we never import
# from the legacy `mediapipe.solutions` API just to borrow a constant.
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),            # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),            # index
    (5, 9), (9, 10), (10, 11), (11, 12),       # middle
    (9, 13), (13, 14), (14, 15), (15, 16),     # ring
    (13, 17), (17, 18), (18, 19), (19, 20),    # pinky
    (0, 17),                                    # palm base
)
