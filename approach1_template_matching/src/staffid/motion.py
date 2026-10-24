"""Moving-person detection by background subtraction (no deep learning needed).

Seated people barely move, walking people do, so foreground blobs of a
person-like size are a cheap proxy for "walking person" boxes.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class MotionParams:
    history: int = 300          # frames remembered by the background model
    var_threshold: float = 40   # MOG2 sensitivity; higher = less noise
    min_area_frac: float = 0.003   # min blob area as a fraction of the frame area
    max_area_frac: float = 0.08    # max blob area (rejects lighting changes)
    morph_kernel: int = 7       # closing kernel joining body parts into one blob
    warmup: int = 50            # frames used to learn the background before detecting


class MotionDetector:
    def __init__(self, params: MotionParams | None = None):
        self.p = params or MotionParams()
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=self.p.history, varThreshold=self.p.var_threshold, detectShadows=True
        )
        k = self.p.morph_kernel
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    def __call__(self, frame: np.ndarray) -> tuple[list[tuple[int, int, int, int]], np.ndarray]:
        """Return (boxes as x, y, w, h, foreground mask) for one frame."""
        fg = self.bg.apply(frame)
        fg = np.where(fg == 255, 255, 0).astype(np.uint8)  # drop shadows (value 127)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        h, w = frame.shape[:2]
        lo, hi = self.p.min_area_frac * h * w, self.p.max_area_frac * h * w
        n, _, stats, _ = cv2.connectedComponentsWithStats(fg)
        boxes = [
            tuple(int(v) for v in stats[i, :4])
            for i in range(1, n)
            if lo <= stats[i, cv2.CC_STAT_AREA] <= hi
        ]
        return boxes, fg
