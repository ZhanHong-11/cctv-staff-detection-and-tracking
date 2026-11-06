"""NCC tag score for every tracked person box in every frame (step 3c detector).

Input: tracks.csv from track_rectified.py. Output: data/tag_scores/ncc_track_scores.csv
(frame, track_id, box, score, raw, twin, match position and template), used by
tag_score_viewer.py (and the step 4 linking, under review). Each box's score uses only
its own frame; the whole video is scored in one go only so the viewer can browse instantly
(see score_tracks_live.py for frame-by-frame scoring). Detector settings = the chosen 3c configuration (`no_mid`: drawn tag,
light + dark looks, verify score).

Usage (from approach2/):
    ../.venv/bin/python scripts/score_tracks.py
    ../.venv/bin/python scripts/score_tracks.py --tracks <tracks.csv> --video <file>
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from staffid2.common import BRIEF_PDF, SAMPLE_VIDEO, TAG_SCORES, TRACKS_CSV, setup_logging  # noqa: E402
from staffid2.tag_score import gray_crops, score_crops  # noqa: E402
from staffid2.ncc import BankConfig, MatchConfig  # noqa: E402
from staffid2.tag_source import draw_tag, find_tag, measure_layout, pdf_jpegs, to_albedo  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", default=str(SAMPLE_VIDEO))
    ap.add_argument("--tracks", default=str(TRACKS_CSV))
    ap.add_argument("--margin", type=float, default=0.15, help="crop margin (as in step 3)")
    ap.add_argument("--looks", nargs="+", default=["light", "dark"], help="template looks (3c: no_mid)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--batch", type=int, default=4000, help="crops held in memory at once")
    ap.add_argument("--out", default=str(TAG_SCORES))
    args = ap.parse_args()

    log = setup_logging("score_tracks")
    log.info("params: %s", vars(args))
    t0 = time.time()
    photo = pdf_jpegs(BRIEF_PDF)[0]
    alb = to_albedo(photo, find_tag(photo))
    drawn = draw_tag(measure_layout(alb), 120)
    albedos, layouts = {"drawn": drawn}, {"drawn": measure_layout(drawn)}
    bcfg, mcfg = replace(BankConfig(), looks=tuple(args.looks)), MatchConfig()
    log.info("bank %s, match %s", bcfg, mcfg)

    tracks = pd.read_csv(args.tracks).reset_index(drop=True)
    parts = []
    for s in range(0, len(tracks), args.batch):
        rows = tracks.iloc[s:s + args.batch]
        crops = gray_crops(args.video, rows, args.margin)
        parts.append(score_crops(crops, albedos, layouts, bcfg, mcfg, args.workers))
        log.info("scored %d / %d boxes (%.0f s)", min(s + args.batch, len(tracks)), len(tracks), time.time() - t0)
    out = pd.concat([tracks, pd.concat(parts, ignore_index=True)], axis=1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.round(4).to_csv(args.out, index=False)
    log.info("wrote %s (%d rows, %.0f s)", args.out, len(out), time.time() - t0)


if __name__ == "__main__":
    main()
