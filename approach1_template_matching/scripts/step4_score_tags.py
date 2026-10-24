"""Step 4A: score every tracked person box for the staff tag (template matching).

Inputs:  data/step2_3/tracks.csv, models/tag_templates/*.png
Outputs: data/step4/tag_scores.csv  frame, track_id, score, tag_x, tag_y (original pixels)
If the ground-truth draft exists, also reports how well the score separates
staff boxes from other people's boxes (per box, i.e. before any track voting).
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
from staffid.common import GT_DIR, MODELS_DIR, ROOT, SAMPLE_VIDEO, iter_frames, setup_logging, video_info  # noqa: E402
from staffid.evaluation import auc, label_boxes_with_gt  # noqa: E402
from staffid.tag import MatchParams, TagMatcher, UprightParams, upright_crop  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=str(SAMPLE_VIDEO))
    ap.add_argument("--tracks-csv", default=str(ROOT / "data" / "step2_3" / "tracks.csv"))
    ap.add_argument("--templates", default=str(MODELS_DIR / "tag_templates"))
    ap.add_argument("--gt", default=str(GT_DIR / "ground_truth_draft.csv"))
    ap.add_argument("--out", default=str(ROOT / "data" / "step4"))
    ap.add_argument("--every", type=int, default=3, help="score every Nth frame (tags change little frame to frame)")
    args = ap.parse_args()

    log = setup_logging("step4_score_tags")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    mp, up = MatchParams(), UprightParams()
    matcher = TagMatcher.from_dir(args.templates, mp, upscale=up.upscale)
    log.info("templates: %d, bank size: %d, match params: %s, upright: %s, every: %d",
             len(matcher.templates), len(matcher.bank), mp, up, args.every)

    tracks = pd.read_csv(args.tracks_csv)
    tracks = tracks[tracks.frame % args.every == 0]
    by_frame = {f: g for f, g in tracks.groupby("frame")}
    rows, t0 = [], time.time()
    for idx, frame in tqdm(iter_frames(args.video), total=video_info(args.video)["frames"], desc="tag score"):
        for r in by_frame.get(idx, pd.DataFrame()).itertuples():
            crop, m = upright_crop(frame, (r.x1, r.y1, r.x2, r.y2), up)
            person_len = max(r.x2 - r.x1, r.y2 - r.y1) * up.upscale
            score, (u, v) = matcher.score(crop, person_len)
            ox, oy = cv2.invertAffineTransform(m) @ np.array([u, v, 1.0])
            rows.append((idx, r.track_id, score, ox, oy))
    elapsed = time.time() - t0
    df = pd.DataFrame(rows, columns=["frame", "track_id", "score", "tag_x", "tag_y"]).round(3)
    df.to_csv(out / "tag_scores.csv", index=False)
    log.info("scored %d boxes in %.1f s (%.1f ms/box)", len(df), elapsed, 1000 * elapsed / max(len(df), 1))

    if Path(args.gt).exists():
        gt = pd.read_csv(args.gt)
        lab = label_boxes_with_gt(tracks.merge(df, on=["frame", "track_id"]), gt)
        lab = lab[lab.certain == 1]
        pos, neg = lab[lab.is_staff == 1].score, lab[lab.is_staff == 0].score
        log.info("boxes in certain frames: staff %d, others %d", len(pos), len(neg))
        for name, s in (("staff", pos), ("others", neg)):
            log.info("%-6s score percentiles 50/90/99: %.3f / %.3f / %.3f", name, *np.percentile(s, [50, 90, 99]))
        log.info("per-box AUC (staff vs others): %.3f", auc(pos.to_numpy(), neg.to_numpy()))
        lab.to_csv(out / "tag_scores_labelled.csv", index=False)


if __name__ == "__main__":
    main()
