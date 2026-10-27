"""Step 1: explore the video and find the walking people.

Runs background subtraction over the whole video, writes every motion box to
data/step1/motion_boxes.csv, and saves zoomed crops of moving people (one every
--crop-every frames) so the tag can be found by eye.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from staffid.common import ROOT, SAMPLE_VIDEO, iter_frames, setup_logging, video_info  # noqa: E402
from staffid.motion import MotionDetector, MotionParams  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=str(SAMPLE_VIDEO))
    ap.add_argument("--out", default=str(ROOT / "data" / "step1"))
    ap.add_argument("--crop-every", type=int, default=10)
    ap.add_argument("--zoom", type=int, default=3)
    args = ap.parse_args()

    log = setup_logging("step1_explore")
    out = Path(args.out)
    (out / "crops").mkdir(parents=True, exist_ok=True)
    info = video_info(args.video)
    log.info("video %s: %s", args.video, info)

    params = MotionParams()
    log.info("motion params: %s", params)
    det = MotionDetector(params)
    rows, frames_with_motion = [], 0
    for idx, frame in iter_frames(args.video):
        boxes, _ = det(frame)
        if idx < params.warmup:
            continue
        frames_with_motion += bool(boxes)
        for j, (x, y, w, h) in enumerate(boxes):
            rows.append((idx, j, x, y, w, h))
            if idx % args.crop_every == 0:
                pad = 10
                crop = frame[max(0, y - pad): y + h + pad, max(0, x - pad): x + w + pad]
                crop = cv2.resize(crop, None, fx=args.zoom, fy=args.zoom, interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(str(out / "crops" / f"f{idx:04d}_b{j}_x{x}_y{y}.jpg"), crop)

    with open(out / "motion_boxes.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["frame", "box", "x", "y", "w", "h"])
        wr.writerows(rows)
    log.info("frames with motion: %d / %d, boxes: %d", frames_with_motion, info["frames"], len(rows))
    log.info("wrote %s and crops/", out / "motion_boxes.csv")


if __name__ == "__main__":
    main()
