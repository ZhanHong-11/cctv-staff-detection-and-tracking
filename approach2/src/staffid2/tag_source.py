"""Where the synthetic tags come from: the brief's reference photo (green shirt), found and cut
out automatically, and a clean drawn tag with the same layout.

The brief PDF holds three JPEGs: the green-shirt reference photo and two snapshots of the staff
member taken from the sample video. Only the reference photo is used; the snapshots are frames of
the test video itself, so using them would leak the test data into the templates.

Tag layout (seen in the photo): a white card with a thick black band along the top and 3 black
bars hanging from it (like piano keys). A tag is stored as an "albedo" map: float32 in [0, 1],
0 = black ink, 1 = white card, the card filling the whole array.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


def pdf_jpegs(pdf: Path) -> list[np.ndarray]:
    """All JPEG (DCTDecode) images embedded in a PDF, in file order, decoded to BGR."""
    data = Path(pdf).read_bytes()
    out = []
    for m in re.finditer(rb"<<(.{0,800}?)>>\s*stream\r?\n", data, re.S):
        d = m.group(1)
        if b"/Image" not in d or b"/DCTDecode" not in d:
            continue
        raw = data[m.end():data.index(b"endstream", m.end())]
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            out.append(img)
    return out


def find_tag(img: np.ndarray, white_sat_max: int = 60, white_val_min: int = 150,
             min_area: int = 80, min_fill: float = 0.45, aspect: tuple[float, float] = (1.2, 3.5),
             green_margin: int = 8, band_frac: float = 0.6) -> tuple[int, int, int, int]:
    """Box (x1, y1, x2, y2), x2/y2 exclusive, of the tag in a photo of a person wearing it.

    1. White card: bright, unsaturated blobs; keep the best filled, card-shaped one.
    2. The black band on top is not part of the white blob: extend the box upward (and by the
       outline on the other sides) while a row/column is mostly "not shirt", where shirt = pixels
       whose green clearly beats red and blue (the reference shirt is green).
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    white = ((hsv[..., 1] < white_sat_max) & (hsv[..., 2] > white_val_min)).astype(np.uint8)
    n, _, st, _ = cv2.connectedComponentsWithStats(white)
    best, best_score = None, 0.0
    for i in range(1, n):
        x, y, w, h, a = st[i]
        fill = a / (w * h)
        if a >= min_area and fill >= min_fill and aspect[0] <= w / h <= aspect[1] and a * fill > best_score:
            best, best_score = (x, y, x + w, y + h), a * fill
    if best is None:
        raise ValueError("no white card-shaped blob found in the reference photo")
    b, g, r = (img[..., k].astype(int) for k in range(3))
    not_shirt = (g - np.maximum(r, b)) < green_margin
    x1, y1, x2, y2 = best

    def row_ok(y: int) -> bool:
        return 0 <= y < img.shape[0] and not_shirt[y, x1:x2].mean() >= band_frac

    def col_ok(x: int) -> bool:
        return 0 <= x < img.shape[1] and not_shirt[y1:y2, x].mean() >= band_frac

    while row_ok(y1 - 1):
        y1 -= 1
    while row_ok(y2):
        y2 += 1
    while col_ok(x1 - 1):
        x1 -= 1
    while col_ok(x2):
        x2 += 1
    return int(x1), int(y1), int(x2), int(y2)


def to_albedo(img: np.ndarray, box: tuple[int, int, int, int], lo_pct: float = 5, hi_pct: float = 95) -> np.ndarray:
    """Gray tag cut to [0, 1]: the darkest lo_pct % -> 0 (ink), the brightest hi_pct % -> 1."""
    x1, y1, x2, y2 = box
    g = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY).astype(np.float32)
    lo, hi = np.percentile(g, [lo_pct, hi_pct])
    return np.clip((g - lo) / max(hi - lo, 1.0), 0, 1)


@dataclass
class Layout:
    """Tag layout as fractions of the card (x of width, y of height)."""
    aspect: float                               # card width / height
    band: float                                 # height of the black top band
    bar_len: float                              # bars reach this far down (from the top)
    bars: list[tuple[float, float]]             # (left, right) of each bar
    border: float                               # dark outline width (fraction of height)


def measure_layout(alb: np.ndarray, ink: float = 0.5, default_bars: int = 3) -> Layout:
    """Read the layout off an albedo map: band = top rows that are mostly ink; bars = ink runs
    in the column profile just below the band; bar length = how far those columns stay ink."""
    h, w = alb.shape
    rows = (alb < ink).mean(axis=1)
    band = 0
    while band < h // 2 and rows[band] > 0.6:
        band += 1
    band = max(band, 1)
    below = alb[band:band + max(2, (h - band) // 3)]
    cols = (below < ink).mean(axis=0) > 0.5
    runs, x = [], 0
    while x < w:
        if cols[x]:
            s = x
            while x < w and cols[x]:
                x += 1
            if 0 < s and x < w:               # skip the outline at the card's left/right edge
                runs.append((s / w, x / w))
        x += 1
    if not runs:
        runs = [((2 * k + 1) / (2 * default_bars + 1), (2 * k + 2) / (2 * default_bars + 1))
                for k in range(default_bars)]
    bar_cols = np.concatenate([np.arange(int(a * w), max(int(b * w), int(a * w) + 1)) for a, b in runs])
    col_ink = (alb[:, bar_cols] < ink).mean(axis=1)
    y = band
    while y < h and col_ink[y] > 0.5:
        y += 1
    return Layout(aspect=w / h, band=band / h, bar_len=y / h, bars=runs, border=outline(alb, ink) / h)


def outline(alb: np.ndarray, ink: float = 0.5) -> float:
    """Mean width (px) of the dark rim on the left, right and bottom edges of the card."""
    def run(lines: np.ndarray) -> int:
        n = 0
        while n < len(lines) // 4 and (lines[n] < ink).mean() > 0.6:
            n += 1
        return n
    return float(np.mean([run(alb.T), run(alb.T[::-1]), run(alb[::-1])]))


def draw_tag(layout: Layout, height: int = 120) -> np.ndarray:
    """Clean albedo map of the tag at the given height (width from the layout's aspect)."""
    w = int(round(height * layout.aspect))
    alb = np.ones((height, w), np.float32)
    alb[:int(round(layout.band * height))] = 0
    y2 = int(round(layout.bar_len * height))
    for a, b in layout.bars:
        alb[:y2, int(round(a * w)):int(round(b * w))] = 0
    t = int(round(layout.border * height))
    if t:
        alb[-t:], alb[:, :t], alb[:, -t:] = 0, 0, 0
    return alb
