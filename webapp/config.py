"""
Configuration constants for the Drone Detection and Counting System.
"""

import os
from pathlib import Path

# Class definitions
CLASS_NAMES = {0: "Human", 1: "Car"}
CLASS_COLORS_BGR = {0: (68, 68, 255), 1: (68, 204, 0)}
CLASS_COLORS_HEX = {0: "#FF4444", 1: "#44CC00"}

# Default inference settings
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45
DEFAULT_IMGSZ = 640
IMGSZ_OPTIONS = [320, 480, 640, 960, 1280]

# Tracker options
TRACKER_OPTIONS = ["bytetrack.yaml", "botsort.yaml"]

# Model search paths (relative to project root)
MODEL_SEARCH_PATHS = [
    "weights and results/optimised 1280/best.pt",
    "weights and results/baseline 640/best.pt",
    "weights/best.pt",
    "best.pt",
]

# Project root (one level up from webapp/)
PROJECT_ROOT = Path(__file__).parent.parent


def find_model_path():
    """Search for model weights in expected locations."""
    for rel_path in MODEL_SEARCH_PATHS:
        full_path = PROJECT_ROOT / rel_path
        if full_path.exists():
            return str(full_path)
        if os.path.exists(rel_path):
            return rel_path
    return None
