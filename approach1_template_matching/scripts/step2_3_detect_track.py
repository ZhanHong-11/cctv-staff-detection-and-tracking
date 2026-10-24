"""Steps 2-3: detect people in every frame (rotation-merged YOLO) and track them.

Outputs (in --out):
  tracks.csv         frame, track_id, x1, y1, x2, y2, conf
  track_summary.csv  one row per track: first/last frame, length, mean box size
  tracks.mp4         annotated video with track IDs, for checking by eye
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from staffid.common import ROOT, SAMPLE_VIDEO, iter_frames, setup_logging, video_info  # noqa: E402
from staffid.detect import DetectParams, PersonDetector  # noqa: E402
from staffid.track import PersonTracker, TrackParams  # noqa: E402


def colour(tid: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(tid)
    return tuple(int(c) for c in rng.integers(60, 255, 3))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=str(SAMPLE_VIDEO))
    ap.add_argument("--out", default=str(ROOT / "data" / "step2_3"))
    ap.add_argument("--weights", default=str(ROOT / "models" / "yolo11s.pt"))
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()

    log = setup_logging("step2_3_detect_track")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    info = video_info(args.video)
    dp, tp = DetectParams(weights=args.weights), TrackParams()
    log.info("video %s: %s", args.video, info)
    log.info("detect params: %s", dp)
    log.info("track params: %s", tp)

    det, trk = PersonDetector(dp), PersonTracker(tp)
    log.info("device: %s", det.device)
    writer = None
    if not args.no_video:
        writer = cv2.VideoWriter(str(out / "tracks.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                 info["fps"], (info["width"], info["height"]))
    rows, n_dets, t0 = [], 0, time.time()
    for idx, frame in tqdm(iter_frames(args.video), total=info["frames"], desc="detect+track"):
        dets = det(frame)
        n_dets += len(dets)
        for x1, y1, x2, y2, tid, conf in trk(dets):
            rows.append((idx, int(tid), x1, y1, x2, y2, conf))
            if writer is not None:
                c = colour(int(tid))
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), c, 2)
                cv2.putText(frame, str(int(tid)), (int(x1), int(y1) + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
        if writer is not None:
            cv2.putText(frame, f"frame {idx}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            writer.write(frame)
    if writer is not None:
        writer.release()
    elapsed = time.time() - t0

    df = pd.DataFrame(rows, columns=["frame", "track_id", "x1", "y1", "x2", "y2", "conf"]).round(1)
    df.to_csv(out / "tracks.csv", index=False)
    df["w"], df["h"] = df.x2 - df.x1, df.y2 - df.y1
    summary = df.groupby("track_id").agg(first=("frame", "min"), last=("frame", "max"), n=("frame", "size"),
                                         mean_w=("w", "mean"), mean_h=("h", "mean"), mean_conf=("conf", "mean"))
    summary.round(1).to_csv(out / "track_summary.csv")

    log.info("processed %d frames in %.1f s (%.3f s/frame)", info["frames"], elapsed, elapsed / info["frames"])
    log.info("detections: %d (%.1f per frame), tracked boxes: %d, tracks: %d",
             n_dets, n_dets / info["frames"], len(df), len(summary))
    log.info("tracks longer than 25 frames: %d", int((summary.n > 25).sum()))
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
