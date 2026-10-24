"""Step 1: build the per-frame ground-truth draft from approach1_groundtruth/staff_chain.json.

For each frame: staff_present, box (x1, y1, x2, y2), centre (x, y), source
(track / interp / hold / none) and certain (whether it counts in the main metrics).
Also writes an annotated video for checking the draft by eye.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from staffid.common import GT_DIR, PROJECT_ROOT, ROOT, iter_frames, setup_logging, video_info  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain", default=str(GT_DIR / "staff_chain.json"))
    ap.add_argument("--tracks-csv", default=str(ROOT / "data" / "step2_3" / "tracks.csv"))
    ap.add_argument("--out", default=str(GT_DIR / "ground_truth_draft.csv"))
    ap.add_argument("--video-out", default=str(ROOT / "data" / "review" / "ground_truth_draft.mp4"))
    ap.add_argument("--max-interp-gap", type=int, default=10,
                    help="interpolated positions count as certain only in gaps up to this many frames")
    args = ap.parse_args()

    log = setup_logging("step1_build_ground_truth")
    chain = json.loads(Path(args.chain).read_text())
    video = PROJECT_ROOT / chain["video"]
    n = video_info(video)["frames"]
    tracks = pd.read_csv(args.tracks_csv)

    box = np.full((n, 4), np.nan)
    source = np.array(["none"] * n, dtype=object)
    certain = np.zeros(n, bool)
    present = np.zeros(n, int)

    seg_ranges = []
    for s in chain["segments"]:
        t = tracks[(tracks.track_id == s["track"]) & tracks.frame.between(s["start"], s["end"])]
        if t.empty:
            raise ValueError(f"segment {s} has no boxes")
        f = t.frame.to_numpy()
        box[f] = t[["x1", "y1", "x2", "y2"]].to_numpy()
        source[f] = "track"
        certain[s["start"]:s["end"] + 1] = s["certain"]
        seg_ranges.append((int(f.min()), int(f.max()), s["certain"]))

    # interpolate every missing frame between the first and last tracked box
    first, last = seg_ranges[0][0], seg_ranges[-1][1]
    known = np.flatnonzero(~np.isnan(box[:, 0]))
    for c in range(4):
        box[first:last + 1, c] = np.interp(np.arange(first, last + 1), known, box[known, c])
    source[first:last + 1][source[first:last + 1] == "none"] = "interp"
    for (_, e1, c1), (s2, _, c2) in zip(seg_ranges, seg_ranges[1:]):
        certain[e1 + 1:s2] = c1 and c2
    present[first:last + 1] = 1

    # hold the last box until the staff member walks out of view
    ls = chain["last_seen"]
    box[last + 1:ls["frame"] + 1] = box[last]
    source[last + 1:ls["frame"] + 1] = "hold"
    present[last + 1:ls["frame"] + 1] = 1
    certain[last + 1:ls["frame"] + 1] = ls["certain"]
    certain[ls["frame"] + 1:] = chain["after_last_seen"]["certain"]

    # positions in long interpolation gaps are guesses (the person may not walk in a straight line)
    pos_certain = certain & (source == "track")
    tracked = np.flatnonzero(source == "track")
    for a, b in zip(tracked, tracked[1:]):
        if 1 < b - a <= args.max_interp_gap + 1:
            pos_certain[a + 1:b] = certain[a + 1:b]

    df = pd.DataFrame({
        "frame": np.arange(n), "staff_present": present,
        "x": (box[:, 0] + box[:, 2]) / 2, "y": (box[:, 1] + box[:, 3]) / 2,
        "x1": box[:, 0], "y1": box[:, 1], "x2": box[:, 2], "y2": box[:, 3],
        "source": source, "certain": certain.astype(int), "pos_certain": pos_certain.astype(int),
    }).round(1)
    df.loc[df.staff_present == 0, ["x", "y", "x1", "y1", "x2", "y2"]] = np.nan
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    log.info("frames: %d, staff present: %d, certain: %d, position certain: %d",
             n, present.sum(), certain.sum(), pos_certain.sum())
    log.info("sources: %s", df.source.value_counts().to_dict())
    log.info("wrote %s", args.out)

    # annotated review video: green = tracked, yellow = interpolated / held; grey label = uncertain
    Path(args.video_out).parent.mkdir(parents=True, exist_ok=True)
    info = video_info(video)
    wr = cv2.VideoWriter(args.video_out, cv2.VideoWriter_fourcc(*"mp4v"), info["fps"], (info["width"], info["height"]))
    for i, frame in iter_frames(video):
        r = df.iloc[i]
        if r.staff_present:
            col = (0, 255, 0) if r.source == "track" else (0, 255, 255)
            cv2.rectangle(frame, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), col, 2)
            cv2.circle(frame, (int(r.x), int(r.y)), 4, col, -1)
        txt = f"frame {i}  staff={r.staff_present}  {r.source}  {'certain' if r.certain else 'UNCERTAIN'}"
        cv2.putText(frame, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        wr.write(frame)
    wr.release()
    log.info("wrote review video %s", args.video_out)


if __name__ == "__main__":
    main()
