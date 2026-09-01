"""
Fetch the MediaPipe hand-landmark model bundle used by the Tasks API.

The model file is not committed to the repo (it is a ~7.6 MB binary build
artifact). Run this once after installing dependencies:

    .venv\\Scripts\\python scripts\\download_model.py

Exact source URL (also recorded in jarvis_vision/config.py):
    https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
DEST = Path(__file__).resolve().parent.parent / "jarvis_vision" / "hand_landmarker.task"


def main() -> int:
    if DEST.exists():
        print(f"already present: {DEST} ({DEST.stat().st_size / 1_048_576:.1f} MB)")
        return 0

    DEST.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {MODEL_URL}")
    # Download to a temp path and rename only on success, so an interrupted
    # transfer never leaves a truncated .task file that MediaPipe then fails
    # to load with an opaque error.
    tmp = DEST.with_suffix(DEST.suffix + ".part")
    urllib.request.urlretrieve(MODEL_URL, tmp)
    size = tmp.stat().st_size
    if size < 1_000_000:  # the real bundle is ~7.6 MB; anything tiny is an error page
        tmp.unlink(missing_ok=True)
        print(f"download looks wrong: only {size} bytes, expected ~7.6 MB", file=sys.stderr)
        return 1
    tmp.replace(DEST)
    print(f"saved {DEST} ({DEST.stat().st_size / 1_048_576:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
