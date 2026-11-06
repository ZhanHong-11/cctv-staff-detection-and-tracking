"""Live (frame-by-frame) NCC tag scoring: a first version, to measure what live costs.

The video is read frame by frame. For each frame, the boxes the tracker gives for that frame
(from tracks.csv: ByteTrack only uses past frames, so these are the boxes a live tracker would
hand over at that moment) are scored with FrameScorer, using nothing from other frames. Same
detector and output columns as score_tracks.py, so tag_score_viewer.py can load the result
(--scores) and --compare checks it gives the same scores as the batch version.

Logged: per-frame scoring time (median, p95, max), frames over the real-time budget
(1 / fps = 40 ms at 25 fps) and the frame rate scoring alone would allow. Person detection and
tracking (~48 ms/frame on MPS, track_rectified.py) are not included: they come from tracks.csv.

Not done yet (ideas to make it faster): fewer template sizes chosen from the crop size, scoring
every 2nd-3rd frame, re-checking seated people less often.

Usage (from approach2/):
    ../.venv/bin/python scripts/score_tracks_live.py --threads 4 --compare
    ../.venv/bin/python scripts/score_tracks_live.py --max-frames 200
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from staffid2.common import BRIEF_PDF, ROOT, SAMPLE_VIDEO, TAG_SCORES, TRACKS_CSV, setup_logging  # noqa: E402
from staffid2.ncc import BankConfig, MatchConfig, build_bank  # noqa: E402
from staffid2.person_crop import Lens  # noqa: E402
from staffid2.tag_score import COLUMNS, FrameScorer  # noqa: E402
from staffid2.tag_source import draw_tag, find_tag, measure_layout, pdf_jpegs, to_albedo  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", default=str(SAMPLE_VIDEO))
    ap.add_argument("--tracks", default=str(TRACKS_CSV))
    ap.add_argument("--margin", type=float, default=0.15, help="crop margin (as score_tracks.py)")
    ap.add_argument("--fov-deg", type=float, default=180.0)
    ap.add_argument("--looks", nargs="+", default=["light", "dark"], help="template looks (3c: no_mid)")
    ap.add_argument("--threads", type=int, default=4, help="boxes of one frame matched in parallel (4 fastest here; 7 slower)")
    ap.add_argument("--max-frames", type=int, default=None, help="stop after this many frames")
    ap.add_argument("--compare", action="store_true", help="check against score_tracks.py's saved scores")
    ap.add_argument("--out", default=str(ROOT / "data" / "tag_scores" / "ncc_track_scores_live.csv"))
    args = ap.parse_args()

    log = setup_logging("score_tracks_live")
    log.info("params: %s", vars(args))
    cv2.setNumThreads(1)
    photo = pdf_jpegs(BRIEF_PDF)[0]
    drawn = draw_tag(measure_layout(to_albedo(photo, find_tag(photo))), 120)
    bank = build_bank({"drawn": drawn}, {"drawn": measure_layout(drawn)}, replace(BankConfig(), looks=tuple(args.looks)))
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.max_frames:
        n = min(n, args.max_frames)
    lens = Lens.for_frame(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), args.fov_deg)
    scorer = FrameScorer(bank, MatchConfig(), lens, args.margin, args.threads)
    tracks = pd.read_csv(args.tracks)
    by_frame = {f: g for f, g in tracks.groupby("frame")}
    log.info("%d templates; %d frames at %.0f fps (budget %.0f ms/frame)", len(bank), n, fps, 1000 / fps)

    parts, ms, n_box = [], [], []
    t_all = time.time()
    for f in range(n):
        ok, img = cap.read()
        if not ok:
            log.warning("cannot read frame %d, stopping", f)
            break
        rows = by_frame.get(f)
        if rows is None:
            ms.append(0.0)
            n_box.append(0)
            continue
        t = time.perf_counter()
        res = scorer.score_frame(img, rows[["x1", "y1", "x2", "y2"]].to_numpy(float))
        ms.append((time.perf_counter() - t) * 1000)
        n_box.append(len(rows))
        parts.append(pd.concat([rows.reset_index(drop=True), pd.DataFrame(res, columns=COLUMNS)], axis=1))
        if f % 250 == 0:
            log.info("frame %d: %d boxes, %.0f ms", f, len(rows), ms[-1])
    scorer.close()
    cap.release()

    out = pd.concat(parts, ignore_index=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.round(4).to_csv(args.out, index=False)
    ms_a = np.array(ms)
    budget = 1000 / fps
    log.info("scoring per frame: median %.0f ms, p95 %.0f, max %.0f (boxes median %d, max %d); "
             "%d of %d frames over the %.0f ms budget; scoring alone allows %.1f fps",
             np.median(ms_a), np.percentile(ms_a, 95), ms_a.max(), np.median(n_box), max(n_box),
             int((ms_a > budget).sum()), len(ms_a), budget, 1000 / max(ms_a.mean(), 1e-9))
    log.info("wrote %s (%d rows, %.0f s total)", args.out, len(out), time.time() - t_all)

    if args.compare:
        ref = pd.read_csv(TAG_SCORES)
        m = out.merge(ref, on=["frame", "track_id"], suffixes=("", "_batch"))
        log.info("compare with %s: %d rows matched, max |score difference| %.5f", TAG_SCORES, len(m),
                 float((m.score - m.score_batch).abs().max()))


if __name__ == "__main__":
    main()
