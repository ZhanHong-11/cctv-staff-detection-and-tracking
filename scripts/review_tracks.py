"""Make a contact sheet of crops for given track IDs, to check tracks by eye.

Example:
  .venv/bin/python scripts/review_tracks.py --tracks 179 --step 15 --out data/review/t179.jpg
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from staffid.common import ROOT, iter_frames  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=str(ROOT / "AI Test Candidates.mp4"))
    ap.add_argument("--tracks-csv", default=str(ROOT / "data" / "step2_3" / "tracks.csv"))
    ap.add_argument("--tracks", type=int, nargs="+", required=True)
    ap.add_argument("--step", type=int, default=10, help="show every Nth frame of each track")
    ap.add_argument("--tile", type=int, default=160)
    ap.add_argument("--cols", type=int, default=12)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.tracks_csv)
    df = df[df.track_id.isin(args.tracks)]
    df = df[df.groupby("track_id").frame.transform(lambda f: (f - f.min()) % args.step == 0)]
    want = {f: g for f, g in df.groupby("frame")}
    tiles = []
    for idx, frame in iter_frames(args.video):
        if idx not in want:
            continue
        for _, r in want[idx].iterrows():
            pad = 10
            x1, y1 = max(0, int(r.x1) - pad), max(0, int(r.y1) - pad)
            crop = frame[y1:int(r.y2) + pad, x1:int(r.x2) + pad]
            t = cv2.resize(crop, (args.tile, int(args.tile * 1.25)), interpolation=cv2.INTER_CUBIC)
            cv2.putText(t, f"{idx} t{int(r.track_id)}", (2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
            tiles.append(t)
    if not tiles:
        sys.exit("no frames found for those tracks")
    while len(tiles) % args.cols:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i:i + args.cols]) for i in range(0, len(tiles), args.cols)])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, sheet)
    print(f"{len(tiles)} tiles -> {args.out}")


if __name__ == "__main__":
    main()
