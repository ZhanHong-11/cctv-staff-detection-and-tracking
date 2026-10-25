"""Person detection on views: run YOLO on all views of a frame in one batch,
map the boxes back to the original frame and merge duplicates with NMS."""
from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
import torch
from torchvision.ops import nms
from ultralytics import YOLO
from ultralytics.utils import SETTINGS

if TYPE_CHECKING:  # only for type hints; detection works on anything with .img and .to_original()
    from .fisheye_view import View

SETTINGS.update({"sync": False})  # no Ultralytics usage analytics

PERSON_CLASS = 0  # COCO class id for "person"
STRIDE = 32       # YOLO input sizes must be multiples of this


def pick_device(name: str = "auto") -> str:
    if name != "auto":
        return name
    return "mps" if torch.backends.mps.is_available() else "cpu"


def yolo_size(views: list[View], scale: float = 1.0) -> int:
    """YOLO input size that keeps the views' own pixel detail (times scale)."""
    long_side = max(max(v.img.shape[:2]) for v in views) * scale
    return int(np.ceil(long_side / STRIDE) * STRIDE)


def detect(model: YOLO, views: list[View], imgsz: int, conf: float, nms_iou: float,
           device: str) -> tuple[np.ndarray, float, float]:
    """Returns merged (N, 6) boxes [x1, y1, x2, y2, conf, view_index] in original
    coordinates, plus YOLO time and map-back + merge time in ms."""
    t0 = time.perf_counter()
    results = model.predict([v.img for v in views], classes=[PERSON_CLASS], conf=conf, imgsz=imgsz,
                            device=device, verbose=False)
    raw = [(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()) for r in results]
    t1 = time.perf_counter()
    parts = [np.hstack([v.to_original(xy), c[:, None], np.full((len(c), 1), k)])
             for k, (v, (xy, c)) in enumerate(zip(views, raw)) if len(xy)]
    if parts:
        boxes = np.vstack(parts).astype(np.float32)
        keep = nms(torch.from_numpy(boxes[:, :4]), torch.from_numpy(boxes[:, 4]), nms_iou).numpy()
        boxes = boxes[keep]
    else:
        boxes = np.zeros((0, 6), np.float32)
    t2 = time.perf_counter()
    return boxes, (t1 - t0) * 1e3, (t2 - t1) * 1e3
