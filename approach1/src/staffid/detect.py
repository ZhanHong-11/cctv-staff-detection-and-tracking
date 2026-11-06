"""Person detection on overhead fisheye frames.

A COCO-trained detector expects upright people. In an overhead fisheye image
people point away from the image centre, so they look upright only near the
top of the frame and upside down near the bottom. We therefore run the
detector on the frame rotated by 0/90/180/270 degrees, rotate the boxes back,
and merge duplicates with non-maximum suppression (NMS).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
from torchvision.ops import nms
from ultralytics import YOLO
from ultralytics.utils import SETTINGS

from .common import MODELS_DIR

SETTINGS.update({"sync": False})  # no anonymous usage analytics from ultralytics


@dataclass
class DetectParams:
    weights: str = str(MODELS_DIR / "yolo11s.pt")
    imgsz: int = 960
    conf: float = 0.25
    nms_iou: float = 0.45
    rotations: tuple[int, ...] = (0, 1, 2, 3)  # multiples of 90 degrees, counter-clockwise
    device: str | None = None  # None = auto (mps / cuda / cpu)


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def unrotate_boxes(xyxy: np.ndarray, k: int, w: int, h: int) -> np.ndarray:
    """Map boxes found in np.rot90(frame, k) back to the original frame (w x h)."""
    x1, y1, x2, y2 = xyxy.T
    k %= 4
    if k == 0:
        pts = (x1, y1, x2, y2)
    elif k == 1:  # rotated 90 deg CCW: rotated (u, v) -> original (w - 1 - v, u)
        pts = (w - y2, x1, w - y1, x2)
    elif k == 2:
        pts = (w - x2, h - y2, w - x1, h - y1)
    else:  # k == 3, 90 deg CW: rotated (u, v) -> original (v, h - 1 - u)
        pts = (y1, h - x2, y2, h - x1)
    return np.stack(pts, axis=1)


class PersonDetector:
    def __init__(self, params: DetectParams | None = None):
        self.p = params or DetectParams()
        self.device = self.p.device or pick_device()
        self.model = YOLO(self.p.weights)

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        """Return an (N, 6) array: x1, y1, x2, y2, confidence, rotation k."""
        h, w = frame.shape[:2]
        imgs = [np.ascontiguousarray(np.rot90(frame, k)) for k in self.p.rotations]
        results = self.model.predict(
            imgs, classes=[0], conf=self.p.conf, imgsz=self.p.imgsz, device=self.device, verbose=False
        )
        all_boxes = []
        for k, r in zip(self.p.rotations, results):
            if len(r.boxes) == 0:
                continue
            xyxy = unrotate_boxes(r.boxes.xyxy.cpu().numpy(), k, w, h)
            conf = r.boxes.conf.cpu().numpy()[:, None]
            all_boxes.append(np.hstack([xyxy, conf, np.full_like(conf, k)]))
        if not all_boxes:
            return np.zeros((0, 6), np.float32)
        boxes = np.vstack(all_boxes).astype(np.float32)
        keep = nms(torch.from_numpy(boxes[:, :4]), torch.from_numpy(boxes[:, 4]), self.p.nms_iou).numpy()
        return boxes[keep]


def draw_boxes(frame: np.ndarray, boxes: np.ndarray, color=(0, 0, 255)) -> np.ndarray:
    out = frame.copy()
    for x1, y1, x2, y2, c, *_ in boxes:
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(out, f"{c:.2f}", (int(x1), int(y1) + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return out
