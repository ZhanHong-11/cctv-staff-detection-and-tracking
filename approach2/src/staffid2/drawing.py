"""Drawing helpers: boxes, headers, image grids, and turning a table of boxes into a video.

The main reusable piece is `table_video`: any table with a `frame` column and box columns
(x1, y1, x2, y2), e.g. data/track_rectified/tracks.csv or ground_truth/ground_truth.csv,
can be drawn onto the original video with a colour and a text per row.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pandas as pd

from .common import open_writer

GREEN, YELLOW, CYAN, MAGENTA, RED = (0, 220, 0), (0, 255, 255), (255, 255, 0), (255, 0, 255), (0, 0, 255)
GREY, WHITE = (150, 150, 150), (255, 255, 255)
# answer-key rows: tracked / interpolated / clicked by the user
SOURCE_COLOR = {"track": GREEN, "interp": YELLOW, "manual": MAGENTA}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def id_colour(i: int) -> tuple[int, int, int]:
    """A fixed, fairly bright colour per ID (same ID -> same colour in every frame and video)."""
    rng = np.random.default_rng(int(i))
    return tuple(int(c) for c in rng.integers(60, 255, 3))


def draw_box(img: np.ndarray, xyxy, color: tuple, text: str | None = None, thickness: int = 2,
             font_scale: float = 0.45, text_inside: bool = False) -> None:
    """Draw one box (in place) with an optional text above it (or inside the top-left corner).
    Boxes containing NaN (no position) are skipped."""
    if np.any(pd.isna(np.asarray(xyxy, dtype=float))):
        return
    x1, y1, x2, y2 = (int(v) for v in xyxy)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if text:
        org = (x1 + 2, y1 + int(30 * font_scale)) if text_inside else (x1, max(y1 - 4, 10))
        cv2.putText(img, text, org, FONT, font_scale, color, max(1, thickness // 2))


def dashed_rect(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, col, th: int, dash: int, gap: int) -> None:
    """Dashed rectangle (in place): dash px drawn, gap px skipped along each edge."""
    for (a, b) in (((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)), ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))):
        n = max(1, int(np.hypot(b[0] - a[0], b[1] - a[1]) // (dash + gap)))
        for k in range(n + 1):
            t0, t1 = k * (dash + gap), k * (dash + gap) + dash
            L = max(np.hypot(b[0] - a[0], b[1] - a[1]), 1)
            if t0 >= L:
                break
            p = lambda t: (int(a[0] + (b[0] - a[0]) * min(t, L) / L), int(a[1] + (b[1] - a[1]) * min(t, L) / L))
            cv2.line(img, p(t0), p(t1), col, th)


def draw_header(img: np.ndarray, text: str, color: tuple = YELLOW) -> None:
    """Text in the top-left corner (in place), e.g. 'frame 700  persons 15'."""
    cv2.putText(img, text, (10, 25), FONT, 0.7, color, 2)


def label(img: np.ndarray, text: str) -> np.ndarray:
    """Copy of img with a caption in the top-left corner (for image grids)."""
    img = img.copy()
    cv2.putText(img, text, (8, 24), FONT, 0.7, YELLOW, 2)
    return img


def mosaic(images: list[np.ndarray], cols: int | None = None) -> np.ndarray:
    """Tile images into a grid, padding each to the largest size."""
    cols = cols or int(np.ceil(np.sqrt(len(images))))
    rows = int(np.ceil(len(images) / cols))
    vh, vw = max(i.shape[0] for i in images), max(i.shape[1] for i in images)
    padded = [cv2.copyMakeBorder(i, 0, vh - i.shape[0], 0, vw - i.shape[1], cv2.BORDER_CONSTANT) for i in images]
    padded += [np.zeros_like(padded[0])] * (rows * cols - len(images))
    return np.vstack([np.hstack(padded[r * cols:(r + 1) * cols]) for r in range(rows)])


def table_video(video: str | Path, table: pd.DataFrame, out: str | Path,
                color: Callable[[pd.Series], tuple] = lambda r: GREEN,
                text: Callable[[pd.Series], str | None] = lambda r: None,
                header: Callable[[int, pd.DataFrame], str] | None = None,
                header_color: Callable[[int, pd.DataFrame], tuple] = lambda i, rows: YELLOW,
                thickness: int = 2, text_inside: bool = False,
                start: int = 0, end: int | None = None) -> int:
    """Draw every row of `table` (columns frame, x1, y1, x2, y2, ...) onto its frame of `video`
    and write frames start..end (default: all) to `out`. color/text are called per row, header
    per frame with that frame's rows (default: 'frame N  boxes K'). Returns frames written."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    wr = open_writer(Path(out), fps, size)
    by_frame = {f: g for f, g in table.groupby("frame")}
    empty = table.iloc[0:0]
    header = header or (lambda i, rows: f"frame {i}  boxes {len(rows)}")
    if start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    i, written = start, 0
    while end is None or i <= end:
        ok, img = cap.read()
        if not ok:
            break
        rows = by_frame.get(i, empty)
        for _, r in rows.iterrows():
            draw_box(img, (r.x1, r.y1, r.x2, r.y2), color(r), text(r), thickness, text_inside=text_inside)
        draw_header(img, header(i, rows), header_color(i, rows))
        wr.write(img)
        i += 1
        written += 1
    cap.release()
    wr.release()
    return written


# ---------------------------------------------------------------- staff following (step 4 / 6)
ORANGE, LIGHT_GREEN = (0, 150, 255), (150, 240, 150)
STATE_COLOR = {"CONFIRMED": GREEN, "UNCERTAIN": LIGHT_GREEN, "FILLED": WHITE, "LOST": RED}


def draw_follow(img: np.ndarray, boxes: pd.DataFrame, targets: pd.DataFrame | None, main_target: int | None,
                hit_recent: set[int], hit_now: set[int]) -> None:
    """Draw one frame of the staff following (in place).
    boxes: this frame's tracked boxes (track_id, x1..y2); targets: this frame's target rows
    (target, state, track_id, x, y, x1..y2, source?, suspects?); main_target: the target the
    main CSV reports (None = staff not present); hit_recent / hit_now: tracks with a tag hit in
    the last focus_time frames / in this frame.
    Colours: grey others, orange hit (recent), yellow suspect, green confirmed, dashed light green
    uncertain, dotted white filled, red cross lost; "STAFF" on the reported target."""
    suspects: set[int] = set()
    if targets is not None and "suspects" in targets:
        for s in targets.suspects.dropna():
            suspects |= {int(x) for x in str(s).split()}
    for _, r in boxes.iterrows():
        tid = int(r.track_id)
        col = YELLOW if tid in suspects else ORANGE if tid in hit_recent else GREY
        cv2.rectangle(img, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), col,
                      2 if tid in suspects or tid in hit_now else 1)
    if targets is None:
        return
    for _, t in targets.iterrows():
        name, col = f"T{int(t.target)}", STATE_COLOR.get(t.state, WHITE)
        if t.state == "LOST":
            cv2.drawMarker(img, (int(t.x), int(t.y)), RED, cv2.MARKER_TILTED_CROSS, 22, 3)
            cv2.putText(img, f"{name} lost", (int(t.x) + 12, int(t.y) - 8), FONT, 0.45, RED, 1)
            continue
        x1, y1, x2, y2 = (int(t[c]) for c in ("x1", "y1", "x2", "y2"))
        if t.state == "CONFIRMED":
            cv2.rectangle(img, (x1, y1), (x2, y2), GREEN, 3)
        elif t.state == "UNCERTAIN":
            dashed_rect(img, x1, y1, x2, y2, LIGHT_GREEN, 2, 8, 6)
        else:
            dashed_rect(img, x1, y1, x2, y2, WHITE, 1, 3, 4)
        is_main = main_target is not None and int(t.target) == main_target
        label = ("STAFF " if is_main else "") + f"{name} {t.state.lower()}"
        cv2.putText(img, label, (x1, max(y1 - 6, 12)), FONT, 0.5, col, 2 if is_main else 1)
