"""Step A of the answer key: rectified YOLO11s detection + ByteTrack over the whole video.

Every frame: 4 rectified views -> YOLO -> boxes mapped back + merged (as in
yolo11_rectified_demo.py) -> ByteTrack, which gives each person an ID that
persists across frames. The tracks are the raw material for the click tool
(step B), where the user picks the staff member's track.

Outputs:
  data/track_rectified/tracks.csv         frame, track_id, x1, y1, x2, y2, conf, x, y  (x, y = box centre)
  data/track_rectified/track_summary.csv  one row per track: first/last frame, frames tracked, gaps
  outputs/track_rectified.mp4             video with track IDs drawn from tracks.csv, to see where tracks break

The per-frame loop is staffid2.pipeline.PersonTracks (also used by staff_detector.py). Since that
move the boxes are rounded from float64 instead of float32: a re-run can differ by 0.1 px on a few
boxes. Do NOT re-run on the sample: the answer key refers to the saved tracks.csv.

Usage (from approach2/):
    ../.venv/bin/python scripts/track_rectified.py
    ../.venv/bin/python scripts/track_rectified.py --video <file> --track-buffer 75
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from staffid2.common import MODELS_DIR, ROOT, SAMPLE_VIDEO, setup_logging  # noqa: E402
from staffid2.detect import pick_device  # noqa: E402
from staffid2.drawing import id_colour, table_video  # noqa: E402
from staffid2.pipeline import PersonTracks, ViewParams  # noqa: E402
from staffid2.track import TrackParams  # noqa: E402


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Per track: span, frames actually tracked, and the gaps inside the span
    (frames where ByteTrack kept the ID alive but output no box)."""
    def one(g: pd.DataFrame) -> pd.Series:
        f = np.sort(g.frame.to_numpy())
        gaps = np.diff(f) - 1
        gaps = gaps[gaps > 0]
        return pd.Series(dict(first=f[0], last=f[-1], span=f[-1] - f[0] + 1, n=len(f),
                              n_gaps=len(gaps), longest_gap=int(gaps.max()) if len(gaps) else 0,
                              mean_conf=g.conf.mean()))
    return df.groupby("track_id").apply(one, include_groups=False).round(2)


def main() -> None:
    tp = TrackParams()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", default=str(SAMPLE_VIDEO))
    ap.add_argument("--weights", default=str(MODELS_DIR / "yolo11s.pt"))
    ap.add_argument("--conf", type=float, default=0.1,
                    help="YOLO minimum confidence; low so ByteTrack's second (weak) matching stage gets boxes")
    ap.add_argument("--nms-iou", type=float, default=0.45)
    ap.add_argument("--fov-deg", type=float, default=180.0)
    ap.add_argument("--n-views", type=int, default=4)
    ap.add_argument("--tilt-deg", type=float, default=45.0)
    ap.add_argument("--view-fov-deg", type=float, default=90.0)
    ap.add_argument("--track-high", type=float, default=tp.track_high_thresh)
    ap.add_argument("--track-low", type=float, default=tp.track_low_thresh)
    ap.add_argument("--new-track", type=float, default=tp.new_track_thresh)
    ap.add_argument("--track-buffer", type=int, default=tp.track_buffer, help="frames a lost track is kept")
    ap.add_argument("--match-thresh", type=float, default=tp.match_thresh)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--data-out", default=str(ROOT / "data" / "track_rectified"))
    ap.add_argument("--video-out", default=str(ROOT / "outputs" / "track_rectified.mp4"))
    ap.add_argument("--no-video", action="store_true", help="skip drawing the video")
    args = ap.parse_args()

    log = setup_logging("track_rectified")
    device = pick_device(args.device)
    tp = TrackParams(track_high_thresh=args.track_high, track_low_thresh=args.track_low,
                     new_track_thresh=args.new_track, track_buffer=args.track_buffer, match_thresh=args.match_thresh)
    log.info("params: %s  device=%s", vars(args), device)
    log.info("track params: %s", tp)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    log.info("video: %s  %dx%d  %.1f fps  %d frames", args.video, w, h, fps, n_frames)

    vp = ViewParams(fov_deg=args.fov_deg, n_views=args.n_views, tilt_deg=args.tilt_deg,
                    view_fov_deg=args.view_fov_deg, conf=args.conf, nms_iou=args.nms_iou)
    people = PersonTracks(args.weights, w, h, device, vp, tp)     # the loop body, shared with staff_detector.py
    log.info("rectified: %d views of %dx%d px", args.n_views, people.size, people.size)
    data_out, video_out = Path(args.data_out), Path(args.video_out)
    data_out.mkdir(parents=True, exist_ok=True)

    parts, idx, t0 = [], 0, time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        parts.append(people(idx, frame))
        if idx % 100 == 0:
            log.info("frame %d/%d  tracked=%d", idx, n_frames, len(parts[-1]))
        idx += 1
    cap.release()
    elapsed = time.time() - t0
    n_dets = people.n_dets

    df = pd.concat(parts, ignore_index=True)
    df.to_csv(data_out / "tracks.csv", index=False)
    summary = summarise(df)
    summary.to_csv(data_out / "track_summary.csv")

    long = summary[summary.n >= fps]  # tracked for at least 1 s
    log.info("processed %d frames in %.1f s (%.1f fps)", idx, elapsed, idx / max(elapsed, 1e-9))
    log.info("detections: %d (%.1f per frame); tracked boxes: %d (%.1f per frame)",
             n_dets, n_dets / idx, len(df), len(df) / idx)
    log.info("tracks: %d total, %d tracked >= 1 s; median length of those %.0f frames",
             len(summary), len(long), long.n.median() if len(long) else 0)
    log.info("gaps inside tracks (ID kept alive by track_buffer): %d, longest %d frames",
             int(summary.n_gaps.sum()), int(summary.longest_gap.max()) if len(summary) else 0)
    log.info("wrote %s, %s", data_out / "tracks.csv", data_out / "track_summary.csv")

    if not args.no_video:
        t1 = time.time()
        table_video(args.video, df, video_out, color=lambda r: id_colour(r.track_id),
                    text=lambda r: str(int(r.track_id)), header=lambda i, rows: f"frame {i}  tracks {len(rows)}",
                    text_inside=True)
        log.info("video drawn from tracks.csv in %.1f s: %s", time.time() - t1, video_out)


if __name__ == "__main__":
    main()
