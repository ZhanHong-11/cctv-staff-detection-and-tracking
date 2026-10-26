"""NCC tag scoring of person crops: in batch (many (frame, box) rows read in one pass over the
video, scored by one process per CPU core) or live (FrameScorer: one frame's boxes at a time,
nothing from other frames)."""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd

from .person_crop import Lens, person_view

from .ncc import BankConfig, MatchConfig, Template, build_bank, match_crop


def gray_crops(video: str, rows: pd.DataFrame, margin: float, fov_deg: float = 180.0) -> list[np.ndarray]:
    """Rectified gray crop for every row (columns frame, x1, y1, x2, y2), in the rows' order."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video}")
    lens = Lens.for_frame(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), fov_deg)
    out: list[np.ndarray | None] = [None] * len(rows)
    order = np.argsort(rows.frame.to_numpy(), kind="stable")
    frames = rows.frame.to_numpy()
    boxes = rows[["x1", "y1", "x2", "y2"]].to_numpy(float)
    i, img = -1, None
    for k in order:
        f = int(frames[k])
        while i < f:
            ok, img = cap.read()
            i += 1
            if not ok:
                raise ValueError(f"cannot read frame {i}")
        view, _ = person_view(img, boxes[k], lens, margin)
        out[k] = cv2.cvtColor(view.img, cv2.COLOR_BGR2GRAY)
    cap.release()
    return out


_BANK: list[Template] = []
_MCFG: MatchConfig | None = None


def _init(albedos: dict, layouts: dict, bcfg: BankConfig, mcfg: MatchConfig) -> None:
    global _BANK, _MCFG
    cv2.setNumThreads(1)
    _BANK, _MCFG = build_bank(albedos, layouts, bcfg), mcfg


COLUMNS = ["score", "raw", "twin", "mx", "my", "template", "t_look", "t_length", "t_angle", "t_flip"]


def _result(gray: np.ndarray, bank: list[Template], mcfg: MatchConfig) -> tuple:
    m = match_crop(gray, bank, mcfg)
    t = bank[m.template] if m.template >= 0 else None
    return (m.score, m.raw, m.twin, m.x, m.y, m.template, t.look if t else "",
            t.length if t else np.nan, t.angle if t else np.nan, t.flip if t else False)


def _score(chunk: list[np.ndarray]) -> list[tuple]:
    return [_result(g, _BANK, _MCFG) for g in chunk]


def score_crops(crops: list[np.ndarray], albedos: dict, layouts: dict, bcfg: BankConfig, mcfg: MatchConfig,
                workers: int | None = None, chunk: int = 64) -> pd.DataFrame:
    """NCC result per crop: score, raw, twin, x, y (crop px), template index, look, length, angle, flip."""
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    chunks = [crops[i:i + chunk] for i in range(0, len(crops), chunk)]
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(albedos, layouts, bcfg, mcfg)) as ex:
        res = [r for part in ex.map(_score, chunks) for r in part]
    return pd.DataFrame(res, columns=COLUMNS)


class FrameScorer:
    """Live scoring: the boxes of one frame at a time, using only that frame (as a live camera
    would, frame by frame). Boxes can be matched in parallel threads (OpenCV releases the GIL)."""

    def __init__(self, bank: list[Template], mcfg: MatchConfig, lens: Lens, margin: float, threads: int = 1):
        self.bank, self.mcfg, self.lens, self.margin = bank, mcfg, lens, margin
        self.pool = ThreadPoolExecutor(threads) if threads > 1 else None

    def _one(self, img: np.ndarray, box) -> tuple:
        view, _ = person_view(img, box, self.lens, self.margin)
        return _result(cv2.cvtColor(view.img, cv2.COLOR_BGR2GRAY), self.bank, self.mcfg)

    def score_frame(self, img: np.ndarray, boxes: np.ndarray) -> list[tuple]:
        """One result tuple (COLUMNS) per box (x1, y1, x2, y2), in order."""
        if self.pool is None:
            return [self._one(img, b) for b in boxes]
        return list(self.pool.map(lambda b: self._one(img, b), boxes))

    def close(self) -> None:
        if self.pool is not None:
            self.pool.shutdown()
