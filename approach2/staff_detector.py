"""Staff detector: find the staff member (white tag with black bars) in an overhead fisheye video
and give their x, y in every frame.

One pass over the video, frame by frame, the way a live camera would run (frame t uses nothing
from later frames):
  1. people: 4 rectified views of the fisheye frame -> YOLO11s -> boxes mapped back and merged
     -> ByteTrack track IDs                                   (src/staffid2/pipeline.py)
  2. tag score of every tracked person (NCC templates drawn from the brief's tag photo)
                                                              (src/staffid2/tag_score.py)
  3. following: no staff until a tag is clearly seen; then follow that person's track, and
     when it is lost, a movement guess plus nearby suspects that the tag can pick
                                                              (src/staffid2/follow.py)
After the last frame, gaps of up to fill_max frames are filled (the saved CSV only).

Writes to <out>/<video name>/:
  staff.csv          frame,staff_present,x,y  (the answer; short gaps filled)
  staff_live.csv     frame,staff_present,x,y  as decided live (no filling)
  detail.csv         staff.csv + target, state, track_id, source per frame
  targets.csv        every target in every frame (state, box, reason, suspects)
  events.csv         what the follower did and why (new target, lost, continuation, suspect hit ...)
  tracks_scored.csv  every tracked person box with its tag score
  run.json           settings, timings, counts (and the score if --answer-key is given)
  staff.mp4          the video with the staff member and the following drawn (unless --no-video)

Settings were found on the sample video (see docs/step4_follow.md); the confirmation threshold
(--conf-thr 0.065) is the main risk on a new video.

Usage (from approach2/):
    ../.venv/bin/python staff_detector.py --video "../AI Test Candidates.mp4"
    ../.venv/bin/python staff_detector.py --video new.mp4 --out result --conf-thr 0.07
    ../.venv/bin/python staff_detector.py --max-frames 200 --answer-key ground_truth/ground_truth.csv
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, fields
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from staffid2.common import MODELS_DIR, ROOT, SAMPLE_VIDEO, open_writer, setup_logging  # noqa: E402
from staffid2.detect import pick_device  # noqa: E402
from staffid2.drawing import CYAN, WHITE, YELLOW, draw_follow  # noqa: E402
from staffid2.follow import FollowConfig, Follower, fill_gaps, main_output, score  # noqa: E402
from staffid2.ncc import MatchConfig  # noqa: E402
from staffid2.person_crop import Lens  # noqa: E402
from staffid2.pipeline import PersonTracks, ViewParams, tag_bank  # noqa: E402
from staffid2.tag_score import COLUMNS, FrameScorer  # noqa: E402
from staffid2.track import TrackParams  # noqa: E402

TARGET_COLUMNS = ["frame", "target", "state", "track_id", "x", "y", "x1", "y1", "x2", "y2", "last_conf", "reason",
                  "suspects"]


def add_dataclass_args(ap: argparse.ArgumentParser, cls, title: str) -> None:
    g = ap.add_argument_group(title)
    for f in fields(cls):
        kind = (lambda v: v.lower() in ("1", "true", "yes")) if isinstance(f.default, bool) else type(f.default)
        g.add_argument("--" + f.name.replace("_", "-"), type=kind, default=f.default)


def from_args(cls, args: argparse.Namespace):
    return cls(**{f.name: getattr(args, f.name) for f in fields(cls)})


def process(args: argparse.Namespace, vp: ViewParams, tp: TrackParams, cfg: FollowConfig, log):
    """The live pass. Returns (tracks with scores, targets, events, timings, video info)."""
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.max_frames:
        n = min(n, args.max_frames)
    device = pick_device(args.device)
    log.info("video: %s  %dx%d  %.1f fps  %d frames; device %s", args.video, w, h, fps, n, device)

    people = PersonTracks(args.weights, w, h, device, vp, tp)
    bank = tag_bank(tuple(args.looks))
    scorer = FrameScorer(bank, MatchConfig(), Lens.for_frame(w, h, vp.fov_deg), args.margin, args.threads)
    follower = Follower(cfg)
    log.info("rectified views %d x %d px; %d tag templates; %d scoring threads", vp.n_views, people.size,
             len(bank), args.threads)

    tracks, targets, ms = [], [], {"people": [], "tags": [], "follow": []}
    t_all = time.time()
    f = 0
    while f < n:
        ok, frame = cap.read()
        if not ok:
            log.warning("cannot read frame %d, stopping", f)
            break
        t0 = time.perf_counter()
        rows = people(f, frame)
        t1 = time.perf_counter()
        if len(rows):
            res = scorer.score_frame(frame, rows[["x1", "y1", "x2", "y2"]].to_numpy(float))
            rows = pd.concat([rows.reset_index(drop=True), pd.DataFrame(res, columns=COLUMNS)], axis=1)
        else:
            rows = rows.assign(**{c: pd.Series(dtype=float) for c in COLUMNS})
        t2 = time.perf_counter()
        targets += follower.step(f, rows)
        t3 = time.perf_counter()
        tracks.append(rows)
        for k, a, b in (("people", t0, t1), ("tags", t1, t2), ("follow", t2, t3)):
            ms[k].append((b - a) * 1000)
        if f % 100 == 0:
            done = time.time() - t_all
            log.info("frame %d/%d: %d people, staff targets %s  (%.1f fps so far)", f, n, len(rows),
                     [f"T{t['target']} {t['state'].lower()}" for t in targets if t["frame"] == f] or "none",
                     (f + 1) / max(done, 1e-9))
        f += 1
    cap.release()
    scorer.close()
    n = f
    total = time.time() - t_all
    timing = {k: round(float(np.median(v)), 1) for k, v in ms.items()}
    log.info("processed %d frames in %.0f s (%.1f fps); median ms per frame: %s", n, total, n / max(total, 1e-9),
             timing)
    tracks = pd.concat(tracks, ignore_index=True)
    events = pd.DataFrame(follower.events, columns=["frame", "target", "event", "track_id", "reason"])
    info = dict(frames=n, fps=fps, width=w, height=h, device=device, seconds=round(total, 1),
                frames_per_second=round(n / max(total, 1e-9), 2), median_ms=timing, detections=people.n_dets,
                tracked_boxes=len(tracks), tracks=int(tracks.track_id.nunique()) if len(tracks) else 0)
    return tracks, pd.DataFrame(targets, columns=TARGET_COLUMNS), events, info


def draw_video(args: argparse.Namespace, out: Path, tracks: pd.DataFrame, targets: pd.DataFrame,
               main: pd.DataFrame, cfg: FollowConfig, fps: float, n: int, log) -> None:
    """Second pass over the video: the saved result drawn on every frame."""
    cap = cv2.VideoCapture(args.video)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    wr = open_writer(out, fps, (w, h))
    boxes = {f: g for f, g in tracks.groupby("frame")}
    tg = {f: g for f, g in targets.groupby("frame")}
    hits = tracks[tracks.score >= cfg.hit_thr]
    hit_frames = {t: np.sort(g.frame.to_numpy()) for t, g in hits.groupby("track_id")}
    empty = tracks.iloc[:0]
    m = main.set_index("frame")
    for f in range(n):
        ok, img = cap.read()
        if not ok:
            break
        b = boxes.get(f, empty)
        recent = {t for t in b.track_id.astype(int) if t in hit_frames and
                  ((hf := hit_frames[t])[(hf <= f) & (hf >= f - cfg.focus_time)]).size}
        now = set(b[b.score >= cfg.hit_thr].track_id.astype(int)) if len(b) else set()
        row = m.loc[f]
        draw_follow(img, b, tg.get(f), int(row.target) if row.staff_present == 1 else None, recent, now)
        text = (f"frame {f}  staff at ({row.x:.0f}, {row.y:.0f})  T{int(row.target)} {str(row.state).lower()}"
                if row.staff_present == 1 else f"frame {f}  staff not found" +
                ("" if pd.isna(row.target) else f" (T{int(row.target)} lost)"))
        cv2.rectangle(img, (0, 0), (w, 26), (0, 0, 0), -1)
        cv2.putText(img, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE if row.staff_present == 1 else CYAN, 1)
        cv2.putText(img, "green confirmed | dashed uncertain | dotted filled | x lost | yellow suspect | orange tag hit",
                    (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, YELLOW, 1)
        wr.write(img)
    wr.release()
    cap.release()
    log.info("wrote %s", out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", default=str(SAMPLE_VIDEO))
    ap.add_argument("--out", default=str(ROOT / "result"), help="a folder per video is made inside it")
    ap.add_argument("--weights", default=str(MODELS_DIR / "yolo11s.pt"))
    ap.add_argument("--device", default="auto", help="auto = Mac GPU (mps) if available, else cpu")
    ap.add_argument("--threads", type=int, default=4, help="tag scoring threads (4 was fastest on the laptop)")
    ap.add_argument("--looks", nargs="+", default=["light", "dark"], help="tag template looks (step 3c: no_mid)")
    ap.add_argument("--margin", type=float, default=0.15, help="person crop margin for the tag score")
    ap.add_argument("--max-frames", type=int, default=None, help="stop after this many frames (quick tests)")
    ap.add_argument("--no-video", action="store_true", help="skip drawing staff.mp4")
    ap.add_argument("--answer-key", default=None, help="optional: score against a key (frame,staff_present,x,y)")
    ap.add_argument("--right-px", type=float, default=25.0, help="scoring: within this of the key = right person")
    add_dataclass_args(ap, ViewParams, "person detection")
    add_dataclass_args(ap, TrackParams, "tracking (ByteTrack)")
    add_dataclass_args(ap, FollowConfig, "following (step 4)")
    args = ap.parse_args()

    log = setup_logging("staff_detector")
    vp, tp, cfg = from_args(ViewParams, args), from_args(TrackParams, args), from_args(FollowConfig, args)
    log.info("params: %s", vars(args))
    out = Path(args.out) / Path(args.video).stem.replace(" ", "_")
    out.mkdir(parents=True, exist_ok=True)

    tracks, targets, events, info = process(args, vp, tp, cfg, log)
    n = info["frames"]
    filled = fill_gaps(targets, cfg)
    live, saved = main_output(targets, n), main_output(filled, n)
    cols = ["frame", "staff_present", "x", "y"]
    saved[cols].to_csv(out / "staff.csv", index=False)
    live[cols].to_csv(out / "staff_live.csv", index=False)
    saved.to_csv(out / "detail.csv", index=False)
    filled.round(1).to_csv(out / "targets.csv", index=False)
    events.to_csv(out / "events.csv", index=False)
    tracks.round(4).to_csv(out / "tracks_scored.csv", index=False)

    conf = events[events.event.isin(["new target", "confirmed", "found again"])]
    info.update(first_confirmation=int(conf.frame.min()) if len(conf) else None,
                targets_created=int((events.event == "new target").sum()),
                frames_with_staff_live=int(live.staff_present.sum()), frames_with_staff=int(saved.staff_present.sum()),
                settings=dict(view=asdict(vp), track=asdict(tp), follow=asdict(cfg), looks=args.looks,
                              margin=args.margin, video=args.video))
    log.info("first confirmation: %s; targets created: %d; staff reported in %d / %d frames (live %d)",
             info["first_confirmation"], info["targets_created"], info["frames_with_staff"], n,
             info["frames_with_staff_live"])
    if info["first_confirmation"] is None:
        log.warning("the tag was never seen clearly (no score > %.3f): no staff reported. If the staff member "
                    "is in the video, try a lower --conf-thr (risk: a wrong person is confirmed)", cfg.conf_thr)
    if args.answer_key:
        key = pd.read_csv(args.answer_key)
        key = key[key.frame < n]
        info["score"] = {"live": score(live, key, events, args.right_px), "saved": score(saved, key, events, args.right_px)}
        for k in ("live", "saved"):
            r = info["score"][k]
            log.info("%s vs key: right when it says staff %d/%d (%.1f %%); right after the first confirmation "
                     "%d/%d (%.1f %%)", k, r["right_when_said"], r["said_staff"], 100 * r["precision"],
                     r["covered_right_after_first"], r["frames_after_first"], 100 * r["coverage_after_first"])
    (out / "run.json").write_text(json.dumps(info, indent=1, default=str))

    if not args.no_video:
        t = time.time()
        draw_video(args, out / "staff.mp4", tracks, filled, saved, cfg, info["fps"], n, log)
        log.info("video drawn in %.0f s", time.time() - t)
    log.info("results in %s", out)


if __name__ == "__main__":
    main()
