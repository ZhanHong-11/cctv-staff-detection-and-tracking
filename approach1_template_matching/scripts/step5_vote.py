"""Steps 5-7: link tracks, vote for staff, output per-frame staff presence + x,y, evaluate.

Inputs:  data/step2_3/tracks.csv, data/step4/tag_scores.csv, approach1_groundtruth/ground_truth_draft.csv
Outputs: data/step5/chains.csv, data/step5/staff_frames.csv, data/step5/metrics.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from staffid.common import GT_DIR, ROOT, SAMPLE_VIDEO, setup_logging, video_info  # noqa: E402
from staffid.evaluation import frame_metrics  # noqa: E402
from staffid.vote import VoteParams, link_tracks, staff_frames, vote  # noqa: E402


def evaluate(pred: pd.DataFrame, gt: pd.DataFrame) -> dict:
    m = pred.merge(gt, on="frame", suffixes=("", "_gt"))
    res = {}
    for name, sel in (("all_frames", np.ones(len(m), bool)), ("certain_frames", m.certain.to_numpy() == 1)):
        res[name] = frame_metrics(m.staff_present.to_numpy()[sel], m.staff_present_gt.to_numpy()[sel])
    both = (m.staff_present == 1) & (m.staff_present_gt == 1) & (m.pos_certain == 1)
    err = np.hypot(m.x[both] - m.x_gt[both], m.y[both] - m.y_gt[both])
    res["position_error_px"] = {"frames": int(both.sum()), "mean": round(float(err.mean()), 1),
                                "median": round(float(err.median()), 1),
                                "within_25px": round(float((err <= 25).mean()), 4),
                                "within_50px": round(float((err <= 50).mean()), 4)}
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=str(SAMPLE_VIDEO))
    ap.add_argument("--tracks-csv", default=str(ROOT / "data" / "step2_3" / "tracks.csv"))
    ap.add_argument("--scores-csv", default=str(ROOT / "data" / "step4" / "tag_scores.csv"))
    ap.add_argument("--gt", default=str(GT_DIR / "ground_truth_draft.csv"))
    ap.add_argument("--out", default=str(ROOT / "data" / "step5"))
    ap.add_argument("--no-link", action="store_true", help="vote per track only (ablation)")
    ap.add_argument("--no-fill", action="store_true", help="do not interpolate inside chains (ablation)")
    args = ap.parse_args()

    log = setup_logging("step5_vote")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    p = VoteParams(fill_gaps=not args.no_fill)
    log.info("vote params: %s, link: %s", p, not args.no_link)
    tracks, scores = pd.read_csv(args.tracks_csv), pd.read_csv(args.scores_csv)
    n = video_info(args.video)["frames"]

    chain = ({int(t): int(t) for t in tracks.track_id.unique()} if args.no_link else link_tracks(tracks, p))
    votes = vote(scores, chain, p)
    log.info("tracks: %d -> chains: %d, staff chains: %d", tracks.track_id.nunique(),
             len(set(chain.values())), int(votes.staff.sum()))
    members = pd.Series(chain).groupby(lambda t: chain[t]).apply(lambda s: list(s.index))
    for c, v in votes[votes.hits > 0].sort_values("hits", ascending=False).iterrows():
        log.info("chain %d: hits %d / %d scored, staff=%s, tracks %s", c, v.hits, v.scored, v.staff,
                 members.get(c, [c]))
    votes.assign(tracks=members.reindex(votes.index).astype(str)).to_csv(out / "chains.csv")

    pred = staff_frames(tracks, chain, votes, n, p, scores)
    pred.round(1).to_csv(out / "staff_frames.csv", index=False)
    log.info("predicted staff frames: %d (track %d, interp %d)", pred.staff_present.sum(),
             (pred.source == "track").sum(), (pred.source == "interp").sum())

    if Path(args.gt).exists():
        res = evaluate(pred, pd.read_csv(args.gt))
        for k, v in res.items():
            log.info("%s: %s", k, v)
        (out / "metrics.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
