"""Person-centred rectified crops: a virtual perspective camera aimed straight at one
person box, so the person comes out upright (outward = up) and undistorted.

    lens = Lens.for_frame(w, h)                      # same lens guess as detection
    view, params = person_view(frame, box, lens)     # view.img = crop, view.to_original(...)

Aim and zoom are computed from the box alone (nothing per frame is chosen by hand):
- azimuth = direction of the box centre seen from the image centre
- tilt    = distance from the image centre / f  (equidistant lens: r = f * theta)
- fov     = angular size of the larger box side, plus `margin`
- size    = native: the crop centre keeps the fisheye's own detail (no up- or down-sampling)
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .fisheye_view import View, lens_focal, native_view_size, rectify_maps


@dataclass(frozen=True)
class Lens:
    """Fisheye lens model (assumption, not calibrated): equidistant, centre + focal in px."""
    cx: float
    cy: float
    f: float
    frame_w: int
    frame_h: int

    @classmethod
    def for_frame(cls, w: int, h: int, fov_deg: float = 180.0, radius: float | None = None) -> Lens:
        """Default guess used by detection: centre = image centre, radius = half the diagonal."""
        radius = radius if radius is not None else float(np.hypot(w, h) / 2)
        return cls(w / 2, h / 2, lens_focal(radius, fov_deg), w, h)


def aim(box, lens: Lens, margin: float) -> dict:
    """Virtual camera parameters for one box (x1, y1, x2, y2) in original-frame pixels."""
    x1, y1, x2, y2 = (float(v) for v in box)
    dx, dy = (x1 + x2) / 2 - lens.cx, (y1 + y2) / 2 - lens.cy
    half = (1 + margin) * max(x2 - x1, y2 - y1) / 2 / lens.f     # angular half-size (rad)
    fov = 2 * float(np.degrees(np.arctan(half)))
    return dict(azimuth_deg=float(np.degrees(np.arctan2(dy, dx))),
                tilt_deg=float(np.degrees(np.hypot(dx, dy) / lens.f)),
                fov_deg=fov, size=max(native_view_size(lens.f, fov), 8))


def person_view(frame: np.ndarray, box, lens: Lens, margin: float = 0.15) -> tuple[View, dict]:
    """Rectified crop around one person box, as a View (so boxes in the crop map back)."""
    p = aim(box, lens, margin)
    mx, my = rectify_maps((lens.cx, lens.cy), lens.f, p["azimuth_deg"], p["tilt_deg"], p["fov_deg"], p["size"])
    img = cv2.remap(frame, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return View(img, "person", map_x=mx, map_y=my, frame_w=lens.frame_w, frame_h=lens.frame_h), p


def original_crop(frame: np.ndarray, box, margin: float = 0.15) -> tuple[np.ndarray, tuple[int, int]]:
    """Plain axis-aligned crop of the original frame (box + margin); returns crop and its top-left."""
    x1, y1, x2, y2 = (float(v) for v in box)
    h, w = frame.shape[:2]
    mx, my = margin * (x2 - x1) / 2, margin * (y2 - y1) / 2
    ox1, oy1 = int(max(x1 - mx, 0)), int(max(y1 - my, 0))
    ox2, oy2 = int(min(np.ceil(x2 + mx), w)), int(min(np.ceil(y2 + my), h))
    return frame[oy1:oy2, ox1:ox2], (ox1, oy1)


def to_original_points(view: View, pts: np.ndarray) -> np.ndarray:
    """Map (N, 2) sub-pixel points in a lookup-table view to original-frame (x, y)."""
    pts = np.asarray(pts, np.float32).reshape(-1, 1, 2)
    xs = cv2.remap(view.map_x, pts[..., 0], pts[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    ys = cv2.remap(view.map_y, pts[..., 0], pts[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return np.stack([xs.ravel(), ys.ravel()], axis=1)
