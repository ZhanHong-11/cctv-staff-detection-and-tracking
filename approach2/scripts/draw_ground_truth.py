"""Draw the answer key (ground_truth/ground_truth.csv) onto the whole video, to see all
labelling sessions together.

Box colour = where the label came from: green = track, yellow = interpolated,
magenta = clicked by the user. The header shows frame, time, source and position;
"absent" (staff not visible) and "unlabelled" frames have no box.

Usage (from approach2/):
    ../.venv/bin/python scripts/draw_ground_truth.py
    ../.venv/bin/python scripts/draw_ground_truth.py --csv ground_truth/ground_truth_2.csv --out outputs/ground_truth_2.mp4
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from staffid2.common import ROOT, SAMPLE_VIDEO, setup_logging  # noqa: E402
from staffid2.drawing import SOURCE_COLOR, WHITE, table_video  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", default=str(SAMPLE_VIDEO))
    ap.add_argument("--csv", default=str(ROOT / "ground_truth" / "ground_truth.csv"))
    ap.add_argument("--fps", type=float, default=25.0, help="only for the time shown in the header")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "ground_truth.mp4"))
    args = ap.parse_args()

    log = setup_logging("draw_ground_truth")
    log.info("params: %s", vars(args))
    gt = pd.read_csv(args.csv)
    log.info("answer key: %s, %d frames, sources %s", args.csv, len(gt), gt.source.value_counts().to_dict())

    def header(i: int, rows: pd.DataFrame) -> str:
        if rows.empty:
            return f"frame {i}  {i / args.fps:.1f}s"
        r = rows.iloc[0]
        pos = f"  ({r.x:.0f}, {r.y:.0f})" if r.staff_present == 1 else ""
        return f"frame {i}  {i / args.fps:.1f}s  staff: {r.source}{pos}"

    def header_color(i: int, rows: pd.DataFrame) -> tuple:
        return SOURCE_COLOR.get(rows.source.iloc[0], WHITE) if len(rows) else WHITE

    t0 = time.time()
    n = table_video(args.video, gt, args.out, color=lambda r: SOURCE_COLOR.get(r.source, WHITE),
                    header=header, header_color=header_color, thickness=3)
    log.info("wrote %d frames in %.1f s: %s", n, time.time() - t0, args.out)


if __name__ == "__main__":
    main()
