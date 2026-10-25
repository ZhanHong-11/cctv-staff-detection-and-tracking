"""Shared helpers for approach 2: paths, logging setup, video writer."""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]      # approach2/ (logs/, outputs/)
PROJECT_ROOT = ROOT.parent                      # shared by all approaches
SAMPLE_VIDEO = PROJECT_ROOT / "AI Test Candidates.mp4"
MODELS_DIR = PROJECT_ROOT / "models"            # shared YOLO weights


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


def open_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    """H.264 (plays in QuickTime); fall back to mp4v if avc1 is unavailable."""
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"avc1"), fps, size)
    if not wr.isOpened():
        wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    return wr
