"""Step 4.2 + 4.4 (UNDER REVIEW): follow the staff member live, frame by frame, and score it.

Input: the tag score of every tracked box (score_tracks.py; each score uses only its own frame,
so feeding them in frame order is the same as live). The Follower (src/staffid2/follow.py)
sees frame t only after frames 0..t-1 and never looks ahead.

Writes to outputs/follow/:
    staff_live.csv     frame,staff_present,x,y   the live output (no filling)
    staff.csv          frame,staff_present,x,y   the saved output (short gaps filled at the end)
    main_detail.csv    the saved output + target, state, track_id, source per frame
    main_live_detail.csv  the same for the live output
    targets_live.csv   every target in every frame (state, track, box, reason) as decided live
    targets.csv        the same after the end fill (LOST stretches that were filled -> FILLED)
    events.csv         new target / confirmed / lost / continuation / revived / dropped, with reasons
    score.json         the three agreed numbers (live and saved) + wrong-person stretches

Usage (from approach2/):
    ../.venv/bin/python scripts/follow_staff.py
    ../.venv/bin/python scripts/follow_staff.py --conf-thr 0.07 --search-time 50
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import fields
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from staffid2.common import ANSWER_KEY, ROOT, SAMPLE_VIDEO, TAG_SCORES, setup_logging  # noqa: E402
from staffid2.follow import FollowConfig, fill_gaps, main_output, run, score  # noqa: E402

import cv2  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", default=str(TAG_SCORES), help="score_tracks.py (or score_tracks_live.py) output")
    ap.add_argument("--video", default=str(SAMPLE_VIDEO), help="only for the frame count")
    ap.add_argument("--answer-key", default=str(ANSWER_KEY))
    ap.add_argument("--right-px", type=float, default=25.0, help="scoring: within this of the key = right person")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "follow"))
    for f in fields(FollowConfig):
        kind = (lambda v: v.lower() in ("1", "true", "yes")) if isinstance(f.default, bool) else type(f.default)
        ap.add_argument("--" + f.name.replace("_", "-"), type=kind, default=f.default)
    args = ap.parse_args()

    log = setup_logging("follow_staff")
    log.info("params: %s", vars(args))
    cfg = FollowConfig(**{f.name: getattr(args, f.name) for f in fields(FollowConfig)})
    cap = cv2.VideoCapture(args.video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    tracks = pd.read_csv(args.scores)
    n = max(n, int(tracks.frame.max()) + 1)
    log.info("%d scored boxes, %d frames", len(tracks), n)

    t0 = time.perf_counter()
    targets, events = run(tracks, n, cfg)
    ms = (time.perf_counter() - t0) * 1000 / n
    log.info("followed %d frames: %.2f ms per frame (following only, scores precomputed)", n, ms)
    filled = fill_gaps(targets, cfg)
    live, saved = main_output(targets, n), main_output(filled, n)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cols = ["frame", "staff_present", "x", "y"]
    live[cols].to_csv(out / "staff_live.csv", index=False)
    saved[cols].to_csv(out / "staff.csv", index=False)
    saved.to_csv(out / "main_detail.csv", index=False)
    live.to_csv(out / "main_live_detail.csv", index=False)
    targets.round(1).to_csv(out / "targets_live.csv", index=False)
    filled.round(1).to_csv(out / "targets.csv", index=False)
    events.to_csv(out / "events.csv", index=False)

    key = pd.read_csv(args.answer_key)
    res = {"live": score(live, key, events, args.right_px), "saved": score(saved, key, events, args.right_px),
           "settings": vars(cfg), "targets_created": int(events.event.eq("new target").sum())}
    (out / "score.json").write_text(json.dumps(res, indent=1))

    for name in ("live", "saved"):
        r = res[name]
        log.info("%s: first confirmation %s; said staff %d frames, right %d (%.1f %%); coverage after first "
                 "%d / %d (%.1f %%); right over all frames %.1f %%", name, r["first_confirmation"], r["said_staff"],
                 r["right_when_said"], 100 * r["precision"], r["covered_right_after_first"], r["frames_after_first"],
                 100 * r["coverage_after_first"], 100 * r["right_all_frames"])
        log.info("  by state: %s", r["by_state"])
        for s in r["wrong_stretches"]:
            log.info("  wrong: %s", s)
    log.info("events: %s", events.event.value_counts().to_dict())
    for _, e in events[events.event != "confirmed"].iterrows():
        log.info("  %4d T%d %-16s %s", e.frame, e.target, e.event, e.reason)
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
