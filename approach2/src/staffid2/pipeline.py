"""The per-frame pipeline pieces for a new video, in the order a live camera would use them:

    PersonTracks   rectified views -> YOLO -> boxes mapped back + merged -> ByteTrack
                   (the loop of scripts/track_rectified.py, as reusable code)
    tag_bank       the NCC template bank from the brief's tag photo (as score_tracks.py)

Each takes one frame and uses nothing from later frames.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from ultralytics import YOLO

from .common import BRIEF_PDF
from .detect import detect, yolo_size
from .fisheye_view import lens_focal, native_view_size, rectified_maker
from .ncc import BankConfig, Template, build_bank
from .tag_source import draw_tag, find_tag, measure_layout, pdf_jpegs, to_albedo
from .track import PersonTracker, TrackParams

TRACK_COLUMNS = ["frame", "track_id", "x1", "y1", "x2", "y2", "conf", "x", "y"]


@dataclass
class ViewParams:
    fov_deg: float = 180.0        # lens model (a guess: equidistant, centre = image centre)
    n_views: int = 4
    tilt_deg: float = 45.0
    view_fov_deg: float = 90.0
    conf: float = 0.1             # low, so ByteTrack's second (weak) matching stage gets boxes
    nms_iou: float = 0.45


class PersonTracks:
    """One frame in -> that frame's tracked person boxes (same columns as tracks.csv)."""

    def __init__(self, weights: str, w: int, h: int, device: str, vp: ViewParams, tp: TrackParams):
        radius = float(np.hypot(w, h) / 2)
        f_fish = lens_focal(radius, vp.fov_deg)
        self.size = native_view_size(f_fish, vp.view_fov_deg)
        self.maker = rectified_maker(w, h, (w / 2, h / 2), f_fish, vp.n_views, vp.tilt_deg, vp.view_fov_deg, self.size)
        self.model, self.tracker = YOLO(weights), PersonTracker(tp)
        self.vp, self.device, self.imgsz = vp, device, None
        self.n_dets = 0

    def __call__(self, f: int, frame: np.ndarray) -> pd.DataFrame:
        views = self.maker(frame)
        self.imgsz = self.imgsz or yolo_size(views)
        dets, _, _ = detect(self.model, views, self.imgsz, self.vp.conf, self.vp.nms_iou, self.device)
        self.n_dets += len(dets)
        tr = self.tracker(dets).astype(np.float64)   # float64 like tracks.csv read back (same crops, same scores)
        df = pd.DataFrame({"frame": f, "track_id": tr[:, 4].astype(int), "x1": tr[:, 0], "y1": tr[:, 1],
                           "x2": tr[:, 2], "y2": tr[:, 3], "conf": tr[:, 5]})
        df["x"], df["y"] = (df.x1 + df.x2) / 2, (df.y1 + df.y2) / 2
        return df.round(1)[TRACK_COLUMNS]


def tag_bank(looks: tuple[str, ...] = ("light", "dark")) -> list[Template]:
    """NCC templates drawn from the tag in the brief's photo (3c config no_mid by default)."""
    photo = pdf_jpegs(BRIEF_PDF)[0]
    drawn = draw_tag(measure_layout(to_albedo(photo, find_tag(photo))), 120)
    return build_bank({"drawn": drawn}, {"drawn": measure_layout(drawn)}, replace(BankConfig(), looks=tuple(looks)))
