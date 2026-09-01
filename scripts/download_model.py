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
    urllib.request.urlretrieve(MODEL_URL, DEST)
    print(f"saved {DEST} ({DEST.stat().st_size / 1_048_576:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
