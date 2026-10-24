"""Find the staff member (person wearing the staff tag) in an overhead fisheye video.

Usage:
  python detect_staff.py --video "../AI Test Candidates.mp4"
  python detect_staff.py --video test.mp4 --out outputs/test --fast --no-video

Outputs (in --out, default outputs/<video name>/):
  staff_frames.csv  frame, staff_present, x, y (+ box and source: track / interp / none)
  summary.json      staff intervals, counts, run time, parameters
  timeline.png      when the staff member is present
  annotated.mp4     video with the staff box (unless --no-video)
  tracks.csv, tag_scores.csv, chains.csv  intermediate results for debugging
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from staffid.common import ROOT, setup_logging  # noqa: E402
from staffid.detect import DetectParams  # noqa: E402
from staffid.pipeline import PipelineParams, intervals, plot_timeline, run, write_video  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", help="output folder (default outputs/<video name>)")
    ap.add_argument("--weights", default=str(ROOT / "models" / "yolo11s.pt"),
                    help="YOLO weights; downloaded automatically if missing")
    ap.add_argument("--fast", action="store_true", help="smaller detector (yolo11n) for slow laptops")
    ap.add_argument("--every", type=int, default=3, help="score the tag on every Nth frame")
    ap.add_argument("--device", default=None, help="mps / cuda / cpu (default: auto)")
    ap.add_argument("--no-video", action="store_true", help="skip writing the annotated video")
    ap.add_argument("--gt", help="optional ground-truth CSV to evaluate against")
    args = ap.parse_args()

    video = Path(args.video)
    if not video.exists():
        sys.exit(f"video not found: {video}")
    out = Path(args.out) if args.out else ROOT / "outputs" / video.stem.replace(" ", "_")
    out.mkdir(parents=True, exist_ok=True)
    log = setup_logging("detect_staff")

    weights = str(ROOT / "models" / "yolo11n.pt") if args.fast else args.weights
    p = PipelineParams(detect=DetectParams(weights=weights, device=args.device), every=args.every)
    res = run(video, p, log)
    pred, info = res["pred"], res["info"]

    pred[["frame", "staff_present", "x", "y", "x1", "y1", "x2", "y2", "source"]].round(1).to_csv(
        out / "staff_frames.csv", index=False)
    res["tracks"].round(1).to_csv(out / "tracks.csv", index=False)
    res["scores"].round(3).to_csv(out / "tag_scores.csv", index=False)
    res["votes"].to_csv(out / "chains.csv")

    fps = info["fps"]
    spans = [(int(a), int(b)) for a, b in intervals(pred.staff_present.to_numpy())]
    summary = {
        "video": str(video), "frames": info["frames"], "fps": fps,
        "staff_frames": int(pred.staff_present.sum()),
        "staff_intervals": [{"first": a, "last": b, "start_s": round(a / fps, 2), "end_s": round(b / fps, 2)}
                            for a, b in spans],
        "analysis_seconds": round(res["seconds"], 1),
        "params": asdict(p),
    }
    gt = None
    if args.gt:
        from staffid.evaluation import frame_metrics
        gt = pd.read_csv(args.gt)
        m = pred.merge(gt, on="frame", suffixes=("", "_gt"))
        summary["metrics_all_frames"] = frame_metrics(m.staff_present.to_numpy(), m.staff_present_gt.to_numpy())
        if "certain" in m:
            c = m.certain == 1
            summary["metrics_certain_frames"] = frame_metrics(m.staff_present[c].to_numpy(),
                                                              m.staff_present_gt[c].to_numpy())
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    plot_timeline(pred, fps, out / "timeline.png", gt)

    log.info("staff present in %d / %d frames", summary["staff_frames"], info["frames"])
    for s in summary["staff_intervals"]:
        log.info("  frames %d-%d  (%.1f s - %.1f s)", s["first"], s["last"], s["start_s"], s["end_s"])
    for k in ("metrics_all_frames", "metrics_certain_frames"):
        if k in summary:
            log.info("%s: %s", k, summary[k])
    if not args.no_video:
        write_video(video, out / "annotated.mp4", res, p)
    log.info("outputs in %s", out)


if __name__ == "__main__":
    main()
