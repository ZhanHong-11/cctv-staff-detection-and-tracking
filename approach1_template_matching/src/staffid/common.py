"""Shared helpers: logging setup and video iteration."""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]      # this approach's folder (data/, logs/, models/, outputs/)
PROJECT_ROOT = ROOT.parent                      # shared by all approaches
SAMPLE_VIDEO = PROJECT_ROOT / "AI Test Candidates.mp4"
GT_DIR = PROJECT_ROOT / "ground_truth"          # answer key shared by all approaches


def setup_logging(step_name: str, log_dir: Path = ROOT / "logs") -> logging.Logger:
    """Log to console and to logs/<step_name>_<timestamp>.log."""
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    log_path = log_dir / f"{step_name}_{stamp}.log"
    logger = logging.getLogger(step_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_path)):
        h.setFormatter(fmt)
        logger.addHandler(h)
    logger.info("log file: %s", log_path)
    return logger


def video_info(path: str | Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return info


def iter_frames(path: str | Path) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (frame_index, BGR frame) for every frame in the video."""
    cap = cv2.VideoCapture(str(path))
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield idx, frame
        idx += 1
    cap.release()
