"""Tag detection by normalised cross-correlation (NCC) against a bank of synthetic templates.

Each template is a clean synthetic tag (synth_tag.render, no noise) with a small margin of
shirt around it, for one combination of length, thickness, rotation, flip and look. The margin
matters: on a dark shirt the tag is "white comb on black", on a light shirt "dark bars on a
plain background", and the template has to say which.

Per person crop (gray, rectified, upright), templates slide over the chest area only
(`region`, fractions of the crop). OpenCV's TM_CCOEFF_NORMED is used: zero-mean and normalised,
so overall brightness and contrast do not matter, only the pattern. Where the image under the
template is nearly flat (std < min_std) the score is unreliable (division by ~0) and is ignored.

Scores
- raw:  best NCC of any tag template. Problem found in 3c: a plain small logo or blob on a shirt
        also correlates well with a tag template (0.97 on a black polo's logo).
- comb: NCC(tag) minus the best NCC of two "no bars" twins with the same size, angle and look
        (teeth filled with ink = a solid block; teeth filled with white = just the band), at the
        same position. A blob fits a twin as well as the tag (comb ~ 0); a tag whose bars are
        visible fits the tag better (comb > 0). Only positions with NCC(tag) >= raw_floor count.
        Found in 3c: maximised over the whole chest area it picks clutter (on the tag only 38%).
- verify: raw NCC picks the position and template (on the tag 75%), then the twins are checked
        only there: score = raw - twin_weight * twin.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import cv2
import numpy as np

from .synth_tag import TagParams, render
from .tag_source import Layout

# Template looks: (shirt, card, ink). NCC ignores absolute levels, so only the relations matter.
LOOK_LEVELS = {"light": (150.0, 150.0, 70.0), "dark": (10.0, 130.0, 10.0), "mid": (80.0, 160.0, 50.0)}


@dataclass
class Template:
    img: np.ndarray                  # float32
    twins: list[np.ndarray]          # same shape, bars removed
    length: float
    ratio: float
    angle: float
    flip: bool
    look: str
    source: str


@dataclass
class BankConfig:
    lengths: tuple = (6.0, 8.0, 11.0, 14.0, 18.0)
    ratios: tuple = (0.3, 0.5)
    angles: tuple = (-60.0, -40.0, -20.0, 0.0, 20.0, 40.0, 60.0)
    flips: tuple = (False,)
    looks: tuple = ("light", "dark", "mid")
    sources: tuple = ("drawn",)
    blur_sigma: float = 0.6       # px: even a sharp camera spreads a 1-px bar
    margin: int = 2               # px of shirt kept around the tag
    ss: int = 8


@dataclass
class MatchConfig:
    region: tuple = (0.25, 0.10, 0.80, 0.70)   # x1, y1, x2, y2 as fractions of the crop (chest area)
    min_std: float = 6.0                        # gray levels; flatter windows are ignored
    mode: str = "verify"                        # "raw", "comb" or "verify" (see module doc)
    raw_floor: float = 0.5                      # comb: only where the tag itself fits at least this well
    twin_weight: float = 1.0                    # verify: score = raw - twin_weight * twin


def twin_albedos(alb: np.ndarray, layout: Layout) -> list[np.ndarray]:
    """The tag without its comb: the area between the band and the bar ends filled with ink,
    and filled with white (outline kept)."""
    h, w = alb.shape
    y1, y2 = int(round(layout.band * h)), int(round(layout.bar_len * h))
    t = int(round(layout.border * h))
    out = []
    for v in (0.0, 1.0):
        a = alb.copy()
        a[y1:y2, t:w - t] = v
        out.append(a)
    return out


def build_bank(albedos: dict[str, np.ndarray], layouts: dict[str, Layout], cfg: BankConfig) -> list[Template]:
    bank = []
    twins = {s: twin_albedos(albedos[s], layouts[s]) for s in cfg.sources}
    for src, L, r, a, fl, look in itertools.product(cfg.sources, cfg.lengths, cfg.ratios, cfg.angles, cfg.flips,
                                                    cfg.looks):
        s, w, k = LOOK_LEVELS[look]
        p = TagParams(length=L, ratio=r, angle=a, flip=fl, shear=0.0, look=look, shirt=s, card=w, ink=k,
                      blur_sigma=cfg.blur_sigma)
        side = int(np.ceil(L * 1.2)) + 2 * cfg.margin + 4
        img, corners = render(albedos[src], p, size=(side, side), ss=cfg.ss)
        x1, y1 = np.floor(corners.min(axis=0)).astype(int) - cfg.margin
        x2, y2 = np.ceil(corners.max(axis=0)).astype(int) + 1 + cfg.margin
        x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, side), min(y2, side)
        tw = [render(ta, p, size=(side, side), ss=cfg.ss)[0][y1:y2, x1:x2].astype(np.float32) for ta in twins[src]]
        bank.append(Template(img[y1:y2, x1:x2].astype(np.float32), tw, L, r, a, fl, look, src))
    return bank


@dataclass
class Match:
    score: float                 # by MatchConfig.mode
    raw: float = -1.0            # NCC of the tag template at that position
    twin: float = -1.0           # best twin NCC at that position
    x: float = float("nan")      # template centre in crop px
    y: float = float("nan")
    template: int = -1
    per_look: dict = field(default_factory=dict)


def local_std(img: np.ndarray, th: int, tw: int) -> np.ndarray:
    """Std of every th x tw window, same layout as matchTemplate's result."""
    s1 = cv2.integral(img)
    s2 = cv2.integral(img * img)

    def win(s: np.ndarray) -> np.ndarray:
        return s[th:, tw:] - s[:-th, tw:] - s[th:, :-tw] + s[:-th, :-tw]
    n = th * tw
    mean = win(s1) / n
    return np.sqrt(np.maximum(win(s2) / n - mean * mean, 0))


def match_crop(gray: np.ndarray, bank: list[Template], cfg: MatchConfig) -> Match:
    """Best match of any template inside the chest area of one gray crop."""
    h, w = gray.shape
    fx1, fy1, fx2, fy2 = cfg.region
    rx1, ry1, rx2, ry2 = int(fx1 * w), int(fy1 * h), int(np.ceil(fx2 * w)), int(np.ceil(fy2 * h))
    reg = gray[ry1:ry2, rx1:rx2].astype(np.float64)
    reg32 = reg.astype(np.float32)
    best = Match(score=-np.inf)
    std_cache: dict[tuple[int, int], np.ndarray] = {}
    for k, t in enumerate(bank):
        th, tw = t.img.shape
        if th > reg.shape[0] or tw > reg.shape[1]:
            continue
        res = cv2.matchTemplate(reg32, t.img, cv2.TM_CCOEFF_NORMED)
        if (th, tw) not in std_cache:
            std_cache[(th, tw)] = local_std(reg, th, tw)
        flat = std_cache[(th, tw)] < cfg.min_std
        res[flat] = -1.0
        if cfg.mode in ("raw", "verify"):
            sc, twin = res, None
        else:
            twin = np.max([cv2.matchTemplate(reg32, tt, cv2.TM_CCOEFF_NORMED) for tt in t.twins], axis=0)
            sc = np.where(res >= cfg.raw_floor, res - twin, -1.0)
        i = int(np.argmax(sc))
        v = float(sc.flat[i])
        if v > best.per_look.get(t.look, -np.inf):
            best.per_look[t.look] = v
        if v > best.score:
            yy, xx = divmod(i, sc.shape[1])
            best = Match(v, float(res.flat[i]), float(twin.flat[i]) if twin is not None else -1.0,
                         rx1 + xx + (tw - 1) / 2, ry1 + yy + (th - 1) / 2, k, best.per_look)
    if cfg.mode == "verify" and best.template >= 0:
        # check the bars only where raw NCC put the tag: how well do the no-bar twins fit there?
        t = bank[best.template]
        th, tw = t.img.shape
        x0, y0 = int(round(best.x - (tw - 1) / 2)) - rx1, int(round(best.y - (th - 1) / 2)) - ry1
        win = reg32[y0:y0 + th, x0:x0 + tw]
        best.twin = max(float(cv2.matchTemplate(win, tt, cv2.TM_CCOEFF_NORMED)[0, 0]) for tt in t.twins)
        best.score = best.raw - cfg.twin_weight * best.twin
    return best
