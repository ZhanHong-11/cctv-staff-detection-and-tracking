"""Step 3a, part 1: cut person-centred rectified crops of the staff member (from the answer key)
and of other people (negatives, from the tracker), every --every frames.

Writes
- data/tag_crops/staff/f0000.png, data/tag_crops/neg/f0000_t12.png and data/tag_crops/crops.csv
  (frame, role, track_id, source, box in original px, camera aim, crop size)
- outputs/tag_crops/staff_sheet_NN.jpg: contact sheets of the staff crops (rectified), to see at a
  glance where the tag shows (green label = tracked box, yellow = interpolated, magenta = clicked)
- outputs/tag_crops/compare_original_rectified.jpg: a few frames, original crop vs rectified crop

Negatives: up to --neg-per-frame other tracks per sampled frame, skipping any box that overlaps
the staff box (IoU > --max-iou): those are usually duplicate boxes of the staff member.

Usage (from approach2/):
    ../.venv/bin/python scripts/staff_crops.py
    ../.venv/bin/python scripts/staff_crops.py --every 2 --neg-per-frame 5
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from staffid2.common import ROOT, SAMPLE_VIDEO, setup_logging  # noqa: E402
from staffid2.drawing import SOURCE_COLOR, WHITE, mosaic  # noqa: E402
from staffid2.person_crop import Lens, original_crop, person_view  # noqa: E402


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def tile(img: np.ndarray, size: int, text: str, color: tuple) -> np.ndarray:
    """Square tile: crop fitted into size x size (aspect kept), caption at the top."""
    s = size / max(img.shape[:2])
    small = cv2.resize(img, (max(1, round(img.shape[1] * s)), max(1, round(img.shape[0] * s))),
                       interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_NEAREST)
    out = np.zeros((size, size, 3), np.uint8)
    out[:small.shape[0], :small.shape[1]] = small
    cv2.rectangle(out, (0, 0), (size, 18), (0, 0, 0), -1)
    cv2.putText(out, text, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", default=str(SAMPLE_VIDEO))
    ap.add_argument("--answer-key", default=str(ROOT / "ground_truth" / "ground_truth.csv"))
    ap.add_argument("--tracks", default=str(ROOT / "data" / "track_rectified" / "tracks.csv"))
    ap.add_argument("--every", type=int, default=5, help="use every Nth frame")
    ap.add_argument("--margin", type=float, default=0.15, help="extra view around the person box")
    ap.add_argument("--neg-per-frame", type=int, default=3, help="other people per sampled frame")
    ap.add_argument("--max-iou", type=float, default=0.3, help="skip negatives overlapping the staff box more")
    ap.add_argument("--min-side", type=float, default=20.0, help="skip negative boxes smaller than this (px)")
    ap.add_argument("--fov-deg", type=float, default=180.0, help="lens field of view across 2*radius")
    ap.add_argument("--tile", type=int, default=200, help="contact sheet tile size (px)")
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--rows", type=int, default=5)
    ap.add_argument("--compare-n", type=int, default=6, help="frames in the original-vs-rectified figure")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-out", default=str(ROOT / "data" / "tag_crops"))
    ap.add_argument("--fig-out", default=str(ROOT / "outputs" / "tag_crops"))
    args = ap.parse_args()

    log = setup_logging("staff_crops")
    log.info("params: %s", vars(args))
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    data_out, fig_out = Path(args.data_out), Path(args.fig_out)
    for d in (data_out / "staff", data_out / "neg"):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
    fig_out.mkdir(parents=True, exist_ok=True)

    key = pd.read_csv(args.answer_key)
    key = key[(key.staff_present == 1) & key.x1.notna()].set_index("frame")
    tracks = pd.read_csv(args.tracks)
    by_frame = {f: g for f, g in tracks.groupby("frame")}
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {args.video}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    lens = Lens.for_frame(w, h, args.fov_deg)
    log.info("video %s: %d frames %dx%d, lens f=%.1f px/rad", args.video, n, w, h, lens.f)

    rows, tiles, compare = [], [], []
    frames = [f for f in range(0, n, args.every) if f in key.index]
    compare_at = set(np.linspace(0, len(frames) - 1, args.compare_n).astype(int).tolist())
    i = -1
    for k, f in enumerate(frames):
        while i < f:
            ok, frame = cap.read()
            i += 1
            if not ok:
                raise ValueError(f"cannot read frame {i}")
        g = key.loc[f]
        box = (g.x1, g.y1, g.x2, g.y2)
        view, p = person_view(frame, box, lens, args.margin)
        cv2.imwrite(str(data_out / "staff" / f"f{f:04d}.png"), view.img)
        rows.append(dict(frame=f, role="staff", track_id=g.track_id, source=g.source,
                         x1=g.x1, y1=g.y1, x2=g.x2, y2=g.y2, **p))
        tiles.append(tile(view.img, args.tile, f"{f}  {g.source}", SOURCE_COLOR.get(g.source, WHITE)))
        if k in compare_at:
            orig, _ = original_crop(frame, box, args.margin)
            size = 2 * args.tile
            compare.append(np.hstack([tile(orig, size, f"frame {f} original", WHITE),
                                      tile(view.img, size, f"frame {f} rectified", WHITE)]))

        others = by_frame.get(f, tracks.iloc[0:0])
        others = [r for r in others.itertuples()
                  if r.track_id != g.track_id and min(r.x2 - r.x1, r.y2 - r.y1) >= args.min_side
                  and iou((r.x1, r.y1, r.x2, r.y2), box) <= args.max_iou]
        for j in rng.permutation(len(others))[:args.neg_per_frame]:
            r = others[j]
            nview, np_ = person_view(frame, (r.x1, r.y1, r.x2, r.y2), lens, args.margin)
            cv2.imwrite(str(data_out / "neg" / f"f{f:04d}_t{r.track_id}.png"), nview.img)
            rows.append(dict(frame=f, role="neg", track_id=r.track_id, source="track",
                             x1=r.x1, y1=r.y1, x2=r.x2, y2=r.y2, **np_))
    cap.release()

    table = pd.DataFrame(rows)
    table.round(2).to_csv(data_out / "crops.csv", index=False)
    per_sheet = args.cols * args.rows
    for s in range(0, len(tiles), per_sheet):
        sheet = mosaic(tiles[s:s + per_sheet], cols=args.cols)
        cv2.imwrite(str(fig_out / f"staff_sheet_{s // per_sheet + 1:02d}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if compare:
        cv2.imwrite(str(fig_out / "compare_original_rectified.jpg"), mosaic(compare, cols=2),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])

    staff = table[table.role == "staff"]
    log.info("staff crops: %d (sources %s), negatives: %d from %d tracks",
             len(staff), staff.source.value_counts().to_dict(), (table.role == "neg").sum(),
             table[table.role == "neg"].track_id.nunique())
    log.info("staff crop size px: min %d median %d max %d; tilt deg: min %.0f max %.0f",
             staff["size"].min(), staff["size"].median(), staff["size"].max(),
             staff.tilt_deg.min(), staff.tilt_deg.max())
    log.info("wrote %s, %d sheets in %s (%.1f s)", data_out / "crops.csv",
             -(-len(tiles) // per_sheet), fig_out, time.time() - t0)


if __name__ == "__main__":
    main()
