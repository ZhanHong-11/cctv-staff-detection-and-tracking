"""Ways to turn one overhead fisheye frame into "views" that YOLO finds easier,
plus the geometry to map boxes found in a view back to the original frame.

Every view maker is built once per frame size (all lookup tables precomputed;
the camera does not move) and then applied to each frame:

    maker = make_view_maker("quadrants", w, h, ...)
    views = maker(frame)                  # list[View]
    xyxy_original = views[i].to_original(xyxy_in_view)

Methods
- plain        the frame itself (baseline)
- rotations    whole frame rotated 0/90/180/270 deg (approach 1)
- quadrants    4 overlapping quadrants around the centre, each rotated by its own angle
- rectified    virtual perspective cameras tilted away from straight down (needs a lens model)
- cylindrical  the circle unrolled into a panorama strip, cut into overlapping tiles

Lens model (rectified, cylindrical; an assumption, not calibrated): equidistant
fisheye, r = f * theta, with f chosen so that `radius` corresponds to fov_deg / 2.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

QUADRANTS = ("top-left", "top-right", "bottom-left", "bottom-right")
EDGE_SAMPLES = 9  # points per box edge when mapping a box through a lookup table


@dataclass
class View:
    """One image for YOLO, and how its pixels relate to the original frame."""
    img: np.ndarray
    name: str
    affine: np.ndarray | None = None   # 2x3, view (u, v) -> original (x, y)
    map_x: np.ndarray | None = None    # lookup table, view pixel -> original x
    map_y: np.ndarray | None = None
    frame_w: int = 0
    frame_h: int = 0

    def to_original(self, xyxy: np.ndarray) -> np.ndarray:
        """Map (N, 4) boxes in this view to (N, 4) boxes in the original frame.

        Points along each box edge are mapped and the axis-aligned box around
        them is taken (a straight box in a warped view is curved in the original).
        """
        if len(xyxy) == 0:
            return np.zeros((0, 4), np.float32)
        t = np.linspace(0, 1, EDGE_SAMPLES)
        out = []
        for x1, y1, x2, y2 in xyxy:
            us = np.concatenate([x1 + (x2 - x1) * t, np.full_like(t, x2), x2 - (x2 - x1) * t, np.full_like(t, x1)])
            vs = np.concatenate([np.full_like(t, y1), y1 + (y2 - y1) * t, np.full_like(t, y2), y2 - (y2 - y1) * t])
            if self.affine is not None:
                xs = self.affine[0, 0] * us + self.affine[0, 1] * vs + self.affine[0, 2]
                ys = self.affine[1, 0] * us + self.affine[1, 1] * vs + self.affine[1, 2]
            else:
                hh, ww = self.map_x.shape
                ui = np.clip(us, 0, ww - 1).astype(int)
                vi = np.clip(vs, 0, hh - 1).astype(int)
                xs, ys = self.map_x[vi, ui], self.map_y[vi, ui]
            out.append([xs.min(), ys.min(), xs.max(), ys.max()])
        boxes = np.array(out, np.float32)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, self.frame_w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, self.frame_h)
        return boxes


# ---------------------------------------------------------------- geometry helpers

def rotation_affine(w: int, h: int, angle_deg: float) -> tuple[np.ndarray, tuple[int, int]]:
    """Affine (2x3) that rotates a w x h image counter-clockwise by angle_deg,
    with the canvas grown so nothing is cut off. Returns (matrix, (new_w, new_h))."""
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    new_w, new_h = int(round(h * sin + w * cos)), int(round(h * cos + w * sin))
    m[0, 2] += new_w / 2 - w / 2
    m[1, 2] += new_h / 2 - h / 2
    return m, (new_w, new_h)


def rotate(img: np.ndarray, angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Rotate CCW; returns (rotated image, 2x3 affine mapping rotated -> source).
    Multiples of 90 deg use np.rot90 (exact, no resampling)."""
    h, w = img.shape[:2]
    m, size = rotation_affine(w, h, angle_deg)
    if angle_deg % 90 == 0:
        out = np.ascontiguousarray(np.rot90(img, int(angle_deg // 90) % 4))
    else:
        out = cv2.warpAffine(img, m, size, borderMode=cv2.BORDER_CONSTANT)
    return out, cv2.invertAffineTransform(m)


def shift_affine(a: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Add a translation to the output side of an affine (used to undo a crop)."""
    a = a.copy()
    a[0, 2] += dx
    a[1, 2] += dy
    return a


def lens_focal(radius: float, fov_deg: float) -> float:
    """Equidistant fisheye: pixels per radian of viewing angle."""
    return radius / np.deg2rad(fov_deg / 2)


def native_view_size(f_fish: float, view_fov_deg: float) -> int:
    """Rectified view size whose centre has the same detail as the fisheye (f_view == f_fish):
    smaller throws detail away, larger only interpolates."""
    return int(round(2 * f_fish * np.tan(np.deg2rad(view_fov_deg) / 2)))


def rectify_maps(center: tuple[float, float], f_fish: float, azimuth_deg: float, tilt_deg: float,
                 view_fov_deg: float, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Lookup tables for one virtual perspective camera looking tilt_deg away from
    straight down, towards azimuth_deg (image angle: 0 = right, 90 = down).
    Outward = up in the view."""
    cx, cy = center
    phi, tilt = np.deg2rad(azimuth_deg), np.deg2rad(tilt_deg)
    # fisheye camera frame: x = image right, y = image down, z = optical axis (into the room)
    fwd = np.array([np.sin(tilt) * np.cos(phi), np.sin(tilt) * np.sin(phi), np.cos(tilt)])
    outward = np.array([np.cos(phi), np.sin(phi), 0.0])
    up = outward - outward.dot(fwd) * fwd
    up /= np.linalg.norm(up)
    down = -up
    right = np.cross(down, fwd)

    f_view = (size / 2) / np.tan(np.deg2rad(view_fov_deg) / 2)
    u, v = np.meshgrid(np.arange(size) - size / 2 + 0.5, np.arange(size) - size / 2 + 0.5)
    rays = u[..., None] * right + v[..., None] * down + f_view * fwd
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)

    theta = np.arccos(np.clip(rays[..., 2], -1, 1))      # angle from optical axis
    ang = np.arctan2(rays[..., 1], rays[..., 0])          # angle around the centre
    r = f_fish * theta                                    # equidistant fisheye
    return (cx + r * np.cos(ang)).astype(np.float32), (cy + r * np.sin(ang)).astype(np.float32)


def cylindrical_maps(center: tuple[float, float], radius: float, out_w: int,
                     start_deg: float = 90.0, extra_deg: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Lookup tables for the panorama strip (radius rows x out_w columns for 360 deg).

    Column -> angle around the centre from start_deg, clockwise (0 = right, 90 = down).
    extra_deg repeats that many degrees past 360 so a person on the seam is whole.
    Row 0 = rim (r = radius), last row = centre.
    """
    cx, cy = center
    out_h = int(round(radius))
    n_cols = out_w + int(round(out_w * extra_deg / 360))
    phi = np.deg2rad(start_deg) + 2 * np.pi * np.arange(n_cols) / out_w
    r = radius * (1 - np.arange(out_h) / out_h)
    rr, pp = np.meshgrid(r, phi, indexing="ij")
    return (cx + rr * np.cos(pp)).astype(np.float32), (cy + rr * np.sin(pp)).astype(np.float32)


# ---------------------------------------------------------------- view makers

ViewMaker = Callable[[np.ndarray], list[View]]


def plain_maker(w: int, h: int) -> ViewMaker:
    ident = np.array([[1, 0, 0], [0, 1, 0]], np.float64)
    return lambda frame: [View(frame, "plain", affine=ident, frame_w=w, frame_h=h)]


def rotations_maker(w: int, h: int, angles_deg: tuple[float, ...] = (0, 90, 180, 270)) -> ViewMaker:
    def make(frame: np.ndarray) -> list[View]:
        views = []
        for a in angles_deg:
            img, back = rotate(frame, a)
            views.append(View(img, f"rot {a:+.0f}", affine=back, frame_w=w, frame_h=h))
        return views
    return make


def quadrant_boxes(w: int, h: int, center: tuple[float, float], overlap: float) -> dict[str, tuple[int, int, int, int]]:
    """Crop boxes (x1, y1, x2, y2) of the 4 quadrants, each reaching `overlap`
    (fraction of the quadrant size) past the cut lines."""
    cx, cy = int(round(center[0])), int(round(center[1]))
    ox, oy = int(overlap * max(cx, w - cx)), int(overlap * max(cy, h - cy))
    return {
        "top-left": (0, 0, min(cx + ox, w), min(cy + oy, h)),
        "top-right": (max(cx - ox, 0), 0, w, min(cy + oy, h)),
        "bottom-left": (0, max(cy - oy, 0), min(cx + ox, w), h),
        "bottom-right": (max(cx - ox, 0), max(cy - oy, 0), w, h),
    }


def quadrants_maker(w: int, h: int, center: tuple[float, float], overlap: float,
                    angles_deg: tuple[float, float, float, float]) -> ViewMaker:
    """angles_deg: CCW rotation per quadrant, order TL TR BL BR.
    (-45, 45, -135, 135) turns each quadrant's outward diagonal to 'up';
    (0, 0, 180, 180) leaves the top as is and flips the bottom."""
    boxes = quadrant_boxes(w, h, center, overlap)

    def make(frame: np.ndarray) -> list[View]:
        views = []
        for name, a in zip(QUADRANTS, angles_deg):
            x1, y1, x2, y2 = boxes[name]
            img, back = rotate(frame[y1:y2, x1:x2], a)
            views.append(View(img, f"{name} rot {a:+.0f}", affine=shift_affine(back, x1, y1), frame_w=w, frame_h=h))
        return views
    return make


def rectified_maker(w: int, h: int, center: tuple[float, float], f_fish: float, n_views: int,
                    tilt_deg: float, view_fov_deg: float, size: int, start_deg: float = 90.0) -> ViewMaker:
    tables = []
    for k in range(n_views):
        az = start_deg + 360.0 * k / n_views
        tables.append((f"azimuth {az % 360:.0f}", *rectify_maps(center, f_fish, az, tilt_deg, view_fov_deg, size)))

    def make(frame: np.ndarray) -> list[View]:
        return [View(cv2.remap(frame, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT), name,
                     map_x=mx, map_y=my, frame_w=w, frame_h=h) for name, mx, my in tables]
    return make


def cylindrical_maker(w: int, h: int, center: tuple[float, float], radius: float, out_w: int,
                      n_tiles: int, overlap: float, start_deg: float = 90.0) -> ViewMaker:
    """Panorama strip cut into n_tiles, each reaching `overlap` (fraction of a tile)
    into the next; the strip is extended past 360 deg so the last tile wraps."""
    tile_w = out_w / n_tiles
    extra_deg = 360.0 * overlap / n_tiles
    mx_all, my_all = cylindrical_maps(center, radius, out_w, start_deg, extra_deg)
    tables = []
    for k in range(n_tiles):
        c1 = int(round(k * tile_w))
        c2 = min(int(round((k + 1 + overlap) * tile_w)), mx_all.shape[1])
        tables.append((f"tile {k}", np.ascontiguousarray(mx_all[:, c1:c2]), np.ascontiguousarray(my_all[:, c1:c2])))

    def make(frame: np.ndarray) -> list[View]:
        return [View(cv2.remap(frame, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT), name,
                     map_x=mx, map_y=my, frame_w=w, frame_h=h) for name, mx, my in tables]
    return make
