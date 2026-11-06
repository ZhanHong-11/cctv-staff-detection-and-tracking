"""Measurements of one tag patch (a small box in a crop): brightness inside vs around, the
brightness profile across the bars, how sharp it is and which way the bars run.

Used on the user's labelled tags (step 3a) and later to check that synthetic tags measure
like real ones. All inputs are grayscale crops and boxes in crop pixels (x2, y2 exclusive).
"""
from __future__ import annotations

import cv2
import numpy as np


def ring_and_inside(gray: np.ndarray, box: tuple[int, int, int, int], ring: int) -> tuple[np.ndarray, np.ndarray]:
    """Pixels inside the box, and pixels in a `ring`-px band around it (clipped to the crop)."""
    x1, y1, x2, y2 = box
    h, w = gray.shape
    rx1, ry1, rx2, ry2 = max(x1 - ring, 0), max(y1 - ring, 0), min(x2 + ring, w), min(y2 + ring, h)
    mask = np.ones((ry2 - ry1, rx2 - rx1), bool)
    mask[y1 - ry1:y2 - ry1, x1 - rx1:x2 - rx1] = False
    return gray[y1:y2, x1:x2].ravel(), gray[ry1:ry2, rx1:rx2][mask]


def bar_profile(gray: np.ndarray, box: tuple[int, int, int, int], pad: int = 1,
                rows: tuple[float, float] = (0.25, 0.75)) -> np.ndarray:
    """Mean brightness per column across the box (+pad px each side), over the middle rows.
    In a rectified crop the person is upright, so the bars run roughly up-down."""
    x1, y1, x2, y2 = box
    h = y2 - y1
    r1 = y1 + int(np.floor(rows[0] * h))
    r2 = max(y1 + int(np.ceil(rows[1] * h)), r1 + 1)
    c1, c2 = max(x1 - pad, 0), min(x2 + pad, gray.shape[1])
    return gray[r1:r2, c1:c2].astype(float).mean(axis=0)


def count_dips(profile: np.ndarray, min_frac: float, min_abs: float) -> int:
    """Local minima whose prominence (lower of the two highest points on either side, minus the
    minimum) is at least min_frac of the profile's range and at least min_abs gray levels."""
    p = profile
    rng = p.max() - p.min()
    n = 0
    for i in range(1, len(p) - 1):
        if p[i] < p[i - 1] and p[i] <= p[i + 1]:
            prom = min(p[:i].max(), p[i + 1:].max()) - p[i]
            if prom >= max(min_frac * rng, min_abs):
                n += 1
    return n


def best_row(gray: np.ndarray, box: tuple[int, int, int, int], dip_frac: float, dip_abs: float,
             pad: int = 1, rows: tuple[float, float] = (0.25, 0.75)) -> tuple[int, np.ndarray, int]:
    """Single pixel row (in the middle band) that shows the most bars: (row y, profile, n dips).
    Averaging rows smears the bars as soon as the tag is slightly rotated; one row does not."""
    x1, y1, x2, y2 = box
    h = y2 - y1
    r1 = y1 + int(np.floor(rows[0] * h))
    r2 = max(y1 + int(np.ceil(rows[1] * h)), r1 + 1)
    c1, c2 = max(x1 - pad, 0), min(x2 + pad, gray.shape[1])
    best = (r1, gray[r1, c1:c2].astype(float), -1)
    for y in range(r1, r2):
        p = gray[y, c1:c2].astype(float)
        n = count_dips(p, dip_frac, dip_abs)
        if n > best[2]:
            best = (y, p, n)
    return best


def edge_sharpness(profile: np.ndarray) -> float:
    """Largest brightness step between neighbouring columns / the profile's range.
    1 = a full dark-to-bright jump within one pixel; small = soft, blurred edges."""
    rng = profile.max() - profile.min()
    return float(np.abs(np.diff(profile)).max() / rng) if rng > 0 and len(profile) > 1 else 0.0


def tag_shape(gray: np.ndarray, box: tuple[int, int, int, int], shirt: float, noise: float = 8.0) -> dict:
    """Rotation and size of the tag itself (not its axis-aligned box), from the spread of the
    pixels that differ from the shirt (bright tag on a dark shirt, dark bars on a light one).
    angle_deg: long axis, 0 = level, + = clockwise on screen (unstable for near-square tags).
    length/thickness: sqrt(12 * variance) along / across that axis (exact for a uniform bar).
    On a light shirt only the bars differ, so this measures the bar block, not the white card."""
    x1, y1, x2, y2 = box
    patch = gray[y1:y2, x1:x2].astype(float)
    wgt = np.clip(np.abs(patch - shirt) - noise, 0, None)
    if wgt.sum() == 0:
        return dict(angle_deg=float("nan"), length=float("nan"), thickness=float("nan"))
    ys, xs = np.mgrid[0:patch.shape[0], 0:patch.shape[1]]
    s = wgt.sum()
    mx, my = (wgt * xs).sum() / s, (wgt * ys).sum() / s
    cov = np.array([[(wgt * (xs - mx) ** 2).sum(), (wgt * (xs - mx) * (ys - my)).sum()],
                    [(wgt * (xs - mx) * (ys - my)).sum(), (wgt * (ys - my) ** 2).sum()]]) / s
    ev = np.sort(np.linalg.eigvalsh(cov))[::-1]
    angle = 0.5 * np.degrees(np.arctan2(2 * cov[0, 1], cov[0, 0] - cov[1, 1]))
    return dict(angle_deg=float(angle), length=float(np.sqrt(12 * ev[0])),
                thickness=float(np.sqrt(12 * max(ev[1], 0))))


def measure(gray: np.ndarray, box: tuple[int, int, int, int], ring: int = 3,
            dip_frac: float = 0.15, dip_abs: float = 8.0) -> dict:
    inside, around = ring_and_inside(gray, box, ring)
    _, row, n_dips = best_row(gray, box, dip_frac, dip_abs)
    p10, p90 = np.percentile(inside, [10, 90])
    shirt = float(np.median(around))
    return dict(inside_p10=float(p10), inside_p90=float(p90), shirt_median=shirt,
                tag_vs_shirt=float(p90 - shirt),     # >0: white tag stands out from the shirt
                bar_contrast=float(p90 - p10),       # bars vs white inside the tag
                n_dips=n_dips,                       # bars seen in the best single row
                sharpness=edge_sharpness(row),       # steepest step in that row / its range
                **tag_shape(gray, box, shirt))
