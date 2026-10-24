"""End-to-end pipeline: video -> per-frame staff presence and x,y.

Pass 1 (one read of the video): rotation-merged YOLO detection, ByteTrack,
and tag scoring of every `every`-th frame. Then tracklet linking + voting.
Pass 2 (optional): write an annotated video.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from .common import ROOT, iter_frames, video_info
from .detect import DetectParams, PersonDetector
from .tag import MatchParams, TagMatcher, UprightParams, upright_crop
from .track import PersonTracker, TrackParams
from .vote import VoteParams, link_tracks, staff_frames, vote


@dataclass
class PipelineParams:
    detect: DetectParams = field(default_factory=DetectParams)
    track: TrackParams = field(default_factory=TrackParams)
    upright: UprightParams = field(default_factory=UprightParams)
    match: MatchParams = field(default_factory=MatchParams)
    vote: VoteParams = field(default_factory=VoteParams)
    templates: str = str(ROOT / "models" / "tag_templates")
    every: int = 3   # score tags on every Nth frame


def run(video: str | Path, p: PipelineParams, log) -> dict:
    info = video_info(video)
    log.info("video %s: %s", video, info)
    det, trk = PersonDetector(p.detect), PersonTracker(p.track)
    matcher = TagMatcher.from_dir(p.templates, p.match, upscale=p.upright.upscale)
    log.info("device %s, weights %s, templates %d, tag every %d frames",
             det.device, p.detect.weights, len(matcher.templates), p.every)

    track_rows, score_rows, t0 = [], [], time.time()
    for idx, frame in tqdm(iter_frames(video), total=info["frames"], desc="analysing", unit="frame"):
        for x1, y1, x2, y2, tid, conf in trk(det(frame)):
            track_rows.append((idx, int(tid), x1, y1, x2, y2, conf))
            if idx % p.every == 0:
                crop, m = upright_crop(frame, (x1, y1, x2, y2), p.upright)
                s, (u, v) = matcher.score(crop, max(x2 - x1, y2 - y1) * p.upright.upscale)
                ox, oy = cv2.invertAffineTransform(m) @ np.array([u, v, 1.0])
                score_rows.append((idx, int(tid), s, ox, oy))
    t_analyse = time.time() - t0

    tracks = pd.DataFrame(track_rows, columns=["frame", "track_id", "x1", "y1", "x2", "y2", "conf"])
    scores = pd.DataFrame(score_rows, columns=["frame", "track_id", "score", "tag_x", "tag_y"])
    chain = link_tracks(tracks, p.vote) if len(tracks) else {}
    votes = vote(scores, chain, p.vote) if len(scores) else pd.DataFrame(columns=["scored", "hits", "best", "staff"])
    pred = staff_frames(tracks, chain, votes, info["frames"], p.vote, scores)
    log.info("analysis took %.1f s (%.3f s/frame); tracks %d, chains %d, staff chains %d",
             t_analyse, t_analyse / max(info["frames"], 1), tracks.track_id.nunique(),
             len(set(chain.values())), int(votes.staff.sum()) if len(votes) else 0)
    return {"info": info, "tracks": tracks, "scores": scores, "chain": chain, "votes": votes,
            "pred": pred, "seconds": t_analyse}


def intervals(present: np.ndarray) -> list[tuple[int, int]]:
    """Runs of consecutive 1s as (first_frame, last_frame)."""
    d = np.diff(np.concatenate([[0], present.astype(int), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1))


def write_video(video: str | Path, out_path: Path, res: dict, p: PipelineParams) -> None:
    info, pred, tracks, scores = res["info"], res["pred"], res["tracks"], res["scores"]
    by_frame = {f: g for f, g in tracks.groupby("frame")}
    hits = scores[scores.score >= p.vote.hit_thresh]
    hit_by_frame = {f: g for f, g in hits.groupby("frame")}
    size = (info["width"], info["height"])
    # H.264 plays in QuickTime and browsers; fall back to MPEG-4 Part 2 if this OpenCV build lacks it
    wr = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"avc1"), info["fps"], size)
    if not wr.isOpened():
        wr = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), info["fps"], size)
    for idx, frame in tqdm(iter_frames(video), total=info["frames"], desc="writing video", unit="frame"):
        for r in by_frame.get(idx, pd.DataFrame()).itertuples():
            cv2.rectangle(frame, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), (160, 160, 160), 1)
        for r in hit_by_frame.get(idx, pd.DataFrame()).itertuples():
            cv2.circle(frame, (int(r.tag_x), int(r.tag_y)), 8, (0, 0, 255), 2)
        r = pred.iloc[idx]
        if r.staff_present:
            col = (0, 220, 0) if r.source == "track" else (0, 200, 255)
            cv2.rectangle(frame, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), col, 3)
            cv2.circle(frame, (int(r.x), int(r.y)), 5, col, -1)
            cv2.putText(frame, f"STAFF ({int(r.x)},{int(r.y)})", (int(r.x1), max(15, int(r.y1) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
        status = "STAFF PRESENT" if r.staff_present else "no staff"
        cv2.putText(frame, f"frame {idx}  {status}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 220, 0) if r.staff_present else (0, 0, 255), 2)
        wr.write(frame)
    wr.release()


def plot_timeline(pred: pd.DataFrame, fps: float, out_path: Path, gt: pd.DataFrame | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [("predicted", pred)] + ([("ground truth", gt)] if gt is not None else [])
    fig, ax = plt.subplots(figsize=(11, 1.2 + 0.6 * len(rows)))
    colours = {"track": "#2a9d4b", "interp": "#e9a23b", "hold": "#e9a23b"}
    for i, (name, df) in enumerate(reversed(rows)):
        for src, col in colours.items():
            sel = (df.staff_present == 1) & (df.source == src)
            for a, b in intervals(sel.to_numpy()):
                ax.broken_barh([(a, b - a + 1)], (i - 0.35, 0.7), color=col)
        if "certain" in df:
            for a, b in intervals((df.certain == 0).to_numpy()):
                ax.broken_barh([(a, b - a + 1)], (i - 0.35, 0.7), facecolor="none", edgecolor="#888",
                               hatch="///", linewidth=0)
    ax.set_yticks(range(len(rows)), [n for n, _ in reversed(rows)])
    ax.set_xlim(0, len(pred))
    ax.set_xlabel(f"frame ({fps:.0f} fps)")
    from matplotlib.patches import Patch
    handles = [Patch(color="#2a9d4b", label="staff, tracked"), Patch(color="#e9a23b", label="staff, interpolated")]
    if gt is not None:
        handles.append(Patch(facecolor="none", edgecolor="#888", hatch="///", label="GT uncertain"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.45 if gt is None else -0.3),
              ncol=3, frameon=False)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
