"""Multi-person tracking with ByteTrack (from ultralytics) on our own detections.

ultralytics normally runs its tracker inside `model.track()`, but our boxes come from
several fisheye views merged in the original frame, so we feed them to BYTETracker directly.
(Copied from approach 1's src/staffid/track.py.)

ByteTrack uses only boxes and confidences, no appearance: it predicts each track's next
box with a Kalman filter (centre, aspect ratio, height + velocities), matches confident
detections by IoU first, then weaker ones, and keeps a lost track for `track_buffer`
frames so the same ID can resume after a short miss.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
from ultralytics.trackers.byte_tracker import BYTETracker


@dataclass
class TrackParams:
    track_high_thresh: float = 0.25  # detections above this are matched first
    track_low_thresh: float = 0.1    # weaker detections can still extend an existing track
    new_track_thresh: float = 0.3    # confidence needed to start a new track
    track_buffer: int = 50           # frames a lost track is kept (2 s at 25 fps) to survive occlusion
    match_thresh: float = 0.8        # association cost threshold (1 - IoU)
    fuse_score: bool = True


class _Dets:
    """Minimal results-like object that BYTETracker accepts (conf, cls, xywh, boolean indexing)."""

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray):
        self.xyxy = xyxy
        self.conf = conf
        self.cls = np.zeros_like(conf)
        self.xywh = np.column_stack([(xyxy[:, 0] + xyxy[:, 2]) / 2, (xyxy[:, 1] + xyxy[:, 3]) / 2,
                                     xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1]])

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, mask) -> "_Dets":
        return _Dets(self.xyxy[mask], self.conf[mask])


class PersonTracker:
    def __init__(self, params: TrackParams | None = None):
        self.p = params or TrackParams()
        self.tracker = BYTETracker(SimpleNamespace(**self.p.__dict__))

    def __call__(self, dets: np.ndarray) -> np.ndarray:
        """dets: (N, >=5) x1, y1, x2, y2, conf. Returns (M, 6): x1, y1, x2, y2, track_id, conf."""
        if len(dets) == 0:
            out = self.tracker.update(_Dets(np.zeros((0, 4), np.float32), np.zeros(0, np.float32)))
        else:
            out = self.tracker.update(_Dets(dets[:, :4], dets[:, 4]))
        if len(out) == 0:
            return np.zeros((0, 6), np.float32)
        return out[:, [0, 1, 2, 3, 4, 5]].astype(np.float32)
