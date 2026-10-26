"""Synthetic tag images: an albedo tag (tag_source.py) placed on a shirt and degraded like the
camera does, so it looks like a real tag in a rectified person crop.

Everything is grayscale (NCC works on gray). One render:
1. shape:   card length L px (long side), thickness = ratio * L, rotation, optional 180 deg flip,
            small shear (chest not facing the camera squarely)
2. levels:  shirt brightness S, card white W, ink K, chosen per "look":
              light  W close to S (the card disappears, only the ink shows - as in the video)
              dark   S near black, W well above it (a white comb on black - the hoodie case)
              mid    in between (grey / coloured clothes; not in the sample video)
3. sensor:  drawn at `ss` x resolution and averaged down (pixels integrate light), then
            Gaussian blur, motion blur, noise, JPEG.
Ranges come from step 3a (docs/step3a_tag_survey.md), widened on purpose.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

LOOKS = ("light", "dark", "mid")

# Default sampling ranges (uniform unless noted). Measured in 3a -> used here:
#   length 5-19 px -> 4-24; rotation -38..+59 -> +-70. `ratio` is the card's drawn thickness;
#   measured thickness/length comes out ~0.15-0.2 higher than drawn (blur), so 0.15-0.6 covers 0.30-0.86.
#   Card on a dark shirt: real +60..+139 above the shirt -> 60-170.
DEFAULT_RANGES = dict(
    length=(4.0, 24.0),          # px, long side of the card
    ratio=(0.15, 0.6),           # card thickness / length as drawn
    angle=(-70.0, 70.0),         # deg, + = clockwise on screen
    flip_p=0.5,                  # chance the tag is upside down (which way it hangs is unknown)
    shear=(-0.15, 0.15),
    look_p=(0.4, 0.4, 0.2),      # light, dark, mid
    light_shirt=(110.0, 200.0), light_card_minus_shirt=(-25.0, 30.0),
    dark_shirt=(0.0, 40.0), dark_card=(60.0, 170.0),
    mid_shirt=(40.0, 110.0), mid_card_plus=(20.0, 110.0),
    ink_frac=(0.05, 0.5),        # ink = this x card level (light/mid): 1-2 px bars never look black
    blur_sigma=(0.0, 1.2),       # px, Gaussian, after the sensor
    motion_len=(0.0, 4.0),       # px, line blur in a random direction
    motion_p=0.5,
    noise=(0.0, 6.0),            # gray levels
    jpeg_q=(25, 90), jpeg_p=0.7,
)


@dataclass
class TagParams:
    length: float
    ratio: float
    angle: float
    flip: bool
    shear: float
    look: str
    shirt: float
    card: float
    ink: float
    blur_sigma: float = 0.0
    motion_len: float = 0.0
    motion_angle: float = 0.0
    noise: float = 0.0
    jpeg_q: int = 0               # 0 = no JPEG
    extra: dict = field(default_factory=dict)


def levels(look: str, rng: np.random.Generator, r: dict, shirt: float | None = None) -> tuple[float, float, float]:
    """(shirt, card white, ink) for a look; `shirt` fixes the shirt level (e.g. a real background)."""
    u = lambda key: rng.uniform(*r[key])  # noqa: E731
    if look == "light":
        s = shirt if shirt is not None else u("light_shirt")
        w = s + u("light_card_minus_shirt")
        k = w * u("ink_frac")
    elif look == "dark":
        s = shirt if shirt is not None else u("dark_shirt")
        w = max(u("dark_card"), s + 30)
        k = rng.uniform(0, s + 10)
    else:
        s = shirt if shirt is not None else u("mid_shirt")
        w = s + u("mid_card_plus")
        k = w * u("ink_frac")
    return float(s), float(np.clip(w, 0, 255)), float(np.clip(min(k, w - 20), 0, 255))


def sample_params(rng: np.random.Generator, ranges: dict | None = None, shirt: float | None = None,
                  look: str | None = None) -> TagParams:
    r = {**DEFAULT_RANGES, **(ranges or {})}
    look = look or str(rng.choice(LOOKS, p=np.asarray(r["look_p"]) / np.sum(r["look_p"])))
    s, w, k = levels(look, rng, r, shirt)
    return TagParams(
        length=rng.uniform(*r["length"]), ratio=rng.uniform(*r["ratio"]), angle=rng.uniform(*r["angle"]),
        flip=bool(rng.random() < r["flip_p"]), shear=rng.uniform(*r["shear"]), look=look,
        shirt=s, card=w, ink=k, blur_sigma=rng.uniform(*r["blur_sigma"]),
        motion_len=rng.uniform(*r["motion_len"]) if rng.random() < r["motion_p"] else 0.0,
        motion_angle=rng.uniform(0, 180), noise=rng.uniform(*r["noise"]),
        jpeg_q=int(rng.integers(*r["jpeg_q"])) if rng.random() < r["jpeg_p"] else 0)


def card_affine(alb_shape: tuple[int, int], p: TagParams, centre: tuple[float, float], ss: int) -> np.ndarray:
    """2x3 affine: albedo pixel -> supersampled output pixel (card centred on `centre`, given
    in output pixels). The card's long side becomes p.length, its short side p.ratio * length."""
    h0, w0 = alb_shape
    sx, sy = p.length * ss / w0, p.ratio * p.length * ss / h0
    a = np.deg2rad(p.angle + (180.0 if p.flip else 0.0))
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    m = rot @ np.array([[1, p.shear], [0, 1]]) @ np.diag([sx, sy])
    c_out = (np.asarray(centre) + 0.5) * ss - 0.5          # pixel-centre convention
    t = c_out - m @ np.array([(w0 - 1) / 2, (h0 - 1) / 2])
    return np.hstack([m, t[:, None]]).astype(np.float64)


def motion_kernel(length: float, angle_deg: float) -> np.ndarray | None:
    if length < 1.0:
        return None
    k = int(np.ceil(length)) | 1
    ker = np.zeros((k, k), np.float32)
    c = k // 2
    dx, dy = np.cos(np.deg2rad(angle_deg)) * length / 2, np.sin(np.deg2rad(angle_deg)) * length / 2
    cv2.line(ker, (int(round(c - dx)), int(round(c - dy))), (int(round(c + dx)), int(round(c + dy))), 1.0, 1,
             cv2.LINE_AA)
    return ker / ker.sum()


def render(alb: np.ndarray, p: TagParams, background: np.ndarray | None = None, size: tuple[int, int] | None = None,
           centre: tuple[float, float] | None = None, ss: int = 8,
           rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Gray uint8 image with the tag, and the card's 4 corners (output px).

    background: gray image (any dtype) to paste onto; None = flat shirt level p.shirt.
    size: (w, h) of the output when there is no background (default: big enough for the tag).
    """
    rng = rng or np.random.default_rng()
    if background is None:
        side = int(np.ceil(p.length * 1.6)) + 6
        w, h = size or (side, side)
        bg = np.full((h, w), p.shirt, np.float32)
    else:
        bg = background.astype(np.float32)
        h, w = bg.shape
    centre = centre or ((w - 1) / 2, (h - 1) / 2)
    m = card_affine(alb.shape, p, centre, ss)
    big = (w * ss, h * ss)
    tone = cv2.warpAffine(p.ink + (p.card - p.ink) * alb.astype(np.float32), m, big, flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    alpha = cv2.warpAffine(np.ones_like(alb, np.float32), m, big, flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    bg_big = cv2.resize(bg, big, interpolation=cv2.INTER_NEAREST)
    # tone is 0 outside the card, so after warping it is already "alpha * tone" at the edges
    img = cv2.resize(tone + (1 - alpha) * bg_big, (w, h), interpolation=cv2.INTER_AREA)
    # blur only around the tag when pasting onto a real background (the background is already real)
    blurred = img
    if p.blur_sigma > 0.05:
        blurred = cv2.GaussianBlur(blurred, (0, 0), p.blur_sigma)
    ker = motion_kernel(p.motion_len, p.motion_angle)
    if ker is not None:
        blurred = cv2.filter2D(blurred, -1, ker, borderType=cv2.BORDER_REFLECT)
    if background is not None:
        mask = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_AREA)
        mask = cv2.GaussianBlur(cv2.dilate(mask, np.ones((3, 3), np.uint8)), (0, 0), 1.0)
        img = mask * blurred + (1 - mask) * img
    else:
        img = blurred
    if p.noise > 0:
        img = img + rng.normal(0, p.noise, img.shape)
    out = np.clip(np.round(img), 0, 255).astype(np.uint8)
    if p.jpeg_q:
        ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, p.jpeg_q])
        out = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
    h0, w0 = alb.shape
    corners = np.array([[0, 0], [w0 - 1, 0], [w0 - 1, h0 - 1], [0, h0 - 1]], np.float64)
    corners = (corners @ m[:, :2].T + m[:, 2] + 0.5) / ss - 0.5
    return out, corners
