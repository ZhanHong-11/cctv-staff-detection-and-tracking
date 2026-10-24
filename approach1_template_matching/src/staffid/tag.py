"""Staff-tag detection inside a person box.

1. upright_crop(): rotate the person so their head points up. In an overhead
   fisheye, a standing person points radially away from the image centre, so
   the rotation angle follows from the box position alone.
2. TagMatcher: normalised cross-correlation of small tag templates over the
   chest region of the upright crop, at a few scales and small rotations.
   Returns the best score and where it was found.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


TEMPLATE_UPSCALE = 3.0  # templates in models/tag_templates were cut from 3x upright crops


@dataclass
class UprightParams:
    upscale: float = 2.0           # tag is ~13 px wide; upscale so templates have some resolution
    out_w: int = 120               # upright crop size in original pixels (before upscaling)
    out_h: int = 180
    centre: tuple[float, float] | None = None  # fisheye centre; None = image centre


def radial_angle(cx: float, cy: float, centre: tuple[float, float]) -> float:
    """Angle (degrees) to rotate so that the outward radial direction points up."""
    dx, dy = cx - centre[0], cy - centre[1]
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return 0.0
    # direction "up" in image coordinates is (0, -1); we want (dx, dy) to map onto it
    return math.degrees(math.atan2(dx, -dy))


def upright_crop(frame: np.ndarray, box: tuple[float, float, float, float],
                 p: UprightParams = UprightParams()) -> tuple[np.ndarray, np.ndarray]:
    """Return (upright crop, 2x3 affine matrix mapping original -> crop coordinates)."""
    h, w = frame.shape[:2]
    centre = p.centre or (w / 2, h / 2)
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    ang = radial_angle(cx, cy, centre)
    s = p.upscale
    m = cv2.getRotationMatrix2D((cx, cy), ang, s)
    m[0, 2] += p.out_w * s / 2 - cx
    m[1, 2] += p.out_h * s / 2 - cy
    crop = cv2.warpAffine(frame, m, (int(p.out_w * s), int(p.out_h * s)), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT)
    return crop, m


@dataclass
class MatchParams:
    scales: tuple[float, ...] = (0.85, 1.0, 1.18)
    angles: tuple[float, ...] = (-12, 0, 12)
    # chest search window, as fractions of the upright crop (x0, y0, x1, y1); used when the
    # person's size is unknown
    window: tuple[float, float, float, float] = (0.15, 0.1, 0.85, 0.75)
    # chest search window relative to the crop centre, in units of the person's length
    # (max box side): x in centre +- half_w, y in [centre - above, centre + below]
    chest_half_w: float = 0.3
    chest_above: float = 0.35
    chest_below: float = 0.15
    blur: int = 3
    # surround check: a tag is a small patch that contrasts with the clothing all around it
    peaks_per_template: int = 3    # candidate locations kept per template variant
    max_candidates: int = 10       # candidates verified per crop, best NCC first
    ring: float = 1.0              # ring width around the patch, as a fraction of the template size
    min_contrast: float = 35       # |mean(patch) - mean(ring)| in grey levels
    max_ring_outliers: float = 0.12  # fraction of ring pixels as bright (dark clothing) / dark (light clothing) as the patch
    min_light_ring: float = 100      # light clothing: mean ring brightness must be at least this


def surround_check(g: np.ndarray, cx: int, cy: int, tw: int, th: int, p: MatchParams) -> bool:
    """True if the patch at (cx, cy) looks like a tag: compact, and in contrast with a uniform ring around it."""
    rw, rh = int(tw * (0.5 + p.ring)), int(th * (0.5 + p.ring))
    x0, y0, x1, y1 = cx - rw, cy - rh, cx + rw, cy + rh
    if x0 < 0 or y0 < 0 or x1 > g.shape[1] or y1 > g.shape[0]:
        return False
    win = g[y0:y1, x0:x1].astype(np.float32)
    mask = np.ones(win.shape, bool)
    px0, py0 = rw - tw // 2, rh - th // 2
    mask[py0:py0 + th, px0:px0 + tw] = False       # True = ring, False = patch
    patch, ring = win[~mask], win[mask]
    if patch.mean() - ring.mean() >= p.min_contrast:
        # bright tag on dark clothing: the whole patch is brighter; ring must hold few bright pixels
        thr = (patch.mean() + ring.mean()) / 2
        outliers = (ring > thr).mean()
    else:
        # dark bars on light clothing: the white patch blends in, so compare the bars (darkest
        # pixels) with the ring, which must be light and hold few dark pixels
        bars = np.percentile(patch, 10)
        if ring.mean() - bars < p.min_contrast or ring.mean() < p.min_light_ring:
            return False
        thr = (bars + ring.mean()) / 2
        outliers = (ring < thr).mean()
    return bool(outliers <= p.max_ring_outliers)


@dataclass
class TagMatcher:
    templates: list[np.ndarray]
    p: MatchParams = field(default_factory=MatchParams)

    def __post_init__(self):
        self.bank = []
        for t in self.templates:
            for s in self.p.scales:
                ts = cv2.resize(t, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                for a in self.p.angles:
                    if a == 0:
                        self.bank.append(ts)
                        continue
                    hh, ww = ts.shape
                    m = cv2.getRotationMatrix2D((ww / 2, hh / 2), a, 1.0)
                    self.bank.append(cv2.warpAffine(ts, m, (ww, hh), borderMode=cv2.BORDER_REPLICATE))

    @classmethod
    def from_dir(cls, path: str | Path, p: MatchParams | None = None, upscale: float = 2.0) -> "TagMatcher":
        """Load templates, resized from TEMPLATE_UPSCALE to the upscale used for the crops."""
        files = sorted(Path(path).glob("*.png"))
        if not files:
            raise FileNotFoundError(f"no tag templates in {path}")
        k = upscale / TEMPLATE_UPSCALE
        tpls = [cv2.imread(str(f), cv2.IMREAD_GRAYSCALE) for f in files]
        tpls = [cv2.resize(t, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k != 1 else t for t in tpls]
        return cls(tpls, p or MatchParams())

    def chest_window(self, W: int, H: int, person_len: float | None) -> tuple[int, int, int, int]:
        """Search window (x0, y0, x1, y1) in upright-crop pixels; person_len is in crop pixels."""
        if person_len is None:
            x0, y0, x1, y1 = self.p.window
            return int(x0 * W), int(y0 * H), int(x1 * W), int(y1 * H)
        cx, cy, L = W / 2, H / 2, person_len
        return (max(0, int(cx - self.p.chest_half_w * L)), max(0, int(cy - self.p.chest_above * L)),
                min(W, int(cx + self.p.chest_half_w * L)), min(H, int(cy + self.p.chest_below * L)))

    def score(self, upright: np.ndarray, person_len: float | None = None) -> tuple[float, tuple[int, int]]:
        """Best NCC score among candidates that pass the surround check, and its (x, y) in
        upright-crop pixels. Returns (0.0, best raw location) when no candidate passes.
        person_len: the person's length in upright-crop pixels (limits the search to the chest)."""
        g = cv2.cvtColor(upright, cv2.COLOR_BGR2GRAY) if upright.ndim == 3 else upright
        if self.p.blur > 1:
            g = cv2.GaussianBlur(g, (self.p.blur, self.p.blur), 0)
        H, W = g.shape
        ox, oy, x1, y1 = self.chest_window(W, H, person_len)
        roi = g[oy:y1, ox:x1]
        cands = []  # (ncc, cx, cy, tw, th) in upright-crop pixels
        for t in self.bank:
            th, tw = t.shape
            if th >= roi.shape[0] or tw >= roi.shape[1]:
                continue
            r = cv2.matchTemplate(roi, t, cv2.TM_CCOEFF_NORMED)
            for _ in range(self.p.peaks_per_template):
                _, mx, _, (u, v) = cv2.minMaxLoc(r)
                cands.append((mx, ox + u + tw // 2, oy + v + th // 2, tw, th))
                r[max(0, v - th // 2):v + th // 2 + 1, max(0, u - tw // 2):u + tw // 2 + 1] = -1  # suppress
        if not cands:
            return 0.0, (0, 0)
        cands.sort(key=lambda c: -c[0])
        for ncc, cx, cy, tw, th in cands[:self.p.max_candidates]:
            if surround_check(g, cx, cy, tw, th, self.p):
                return float(ncc), (cx, cy)
        return 0.0, (cands[0][1], cands[0][2])
