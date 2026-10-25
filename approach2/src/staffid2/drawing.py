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
