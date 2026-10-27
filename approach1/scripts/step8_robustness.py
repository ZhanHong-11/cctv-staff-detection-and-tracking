"""Step 8: robustness check on altered copies of the sample video.

We only have one real video, so we simulate "unseen" ones: mirrored, rotated
90 degrees, shrunk to 75 %, and darkened. The pipeline runs on each with the
same settings (nothing re-tuned). Predicted positions are mapped back to the
original coordinates and scored against the ground-truth draft.

Note: these are the same people, clothes and tag, so this tests geometry,
scale and lighting changes, not new people or a new room.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from staffid.common import GT_DIR, ROOT, SAMPLE_VIDEO, iter_frames, setup_logging, video_info  # noqa: E402
from staffid.evaluation import frame_metrics  # noqa: E402
from staffid.pipeline import PipelineParams, run  # noqa: E402

VIDEO = SAMPLE_VIDEO
W, H = 960, 720

# name -> (frame transform, inverse point transform back to original coordinates)
VARIANTS = {
    "hflip": (lambda f: cv2.flip(f, 1), lambda x, y: (W - 1 - x, y)),
    "rot90": (lambda f: cv2.rotate(f, cv2.ROTATE_90_CLOCKWISE), lambda x, y: (y, H - 1 - x)),
    "scale75": (lambda f: cv2.resize(f, (int(W * 0.75), int(H * 0.75)), interpolation=cv2.INTER_AREA),
                lambda x, y: (x / 0.75, y / 0.75)),
    "dark": (lambda f: np.clip(f.astype(np.float32) * 0.55, 0, 255).astype(np.uint8), lambda x, y: (x, y)),
}


def make_variant(name: str, out: Path) -> Path:
    path = out / f"{name}.mp4"
    if path.exists():
        return path
    tf = VARIANTS[name][0]
    info = video_info(VIDEO)
    first = tf(np.zeros((H, W, 3), np.uint8))
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), info["fps"], (first.shape[1], first.shape[0]))
    for _, f in iter_frames(VIDEO):
        wr.write(tf(f))
    wr.release()
    return path


def score(pred: pd.DataFrame, gt: pd.DataFrame, inv) -> dict:
    x, y = inv(pred.x.to_numpy(), pred.y.to_numpy())
    pred = pred.assign(x=x, y=y)
    m = pred.merge(gt, on="frame", suffixes=("", "_gt"))
    c = m.certain == 1
    res = frame_metrics(m.staff_present[c].to_numpy(), m.staff_present_gt[c].to_numpy())
    both = (m.staff_present == 1) & (m.staff_present_gt == 1) & (m.pos_certain == 1)
    err = np.hypot(m.x[both] - m.x_gt[both], m.y[both] - m.y_gt[both])
    res["within_25px"] = round(float((err <= 25).mean()), 4) if both.any() else None
    res["median_err_px"] = round(float(err.median()), 1) if both.any() else None
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--out", default=str(ROOT / "data" / "robustness"))
    args = ap.parse_args()

    log = setup_logging("step8_robustness")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    gt = pd.read_csv(GT_DIR / "ground_truth_draft.csv")
    results = {}
    for name in args.variants:
        path = make_variant(name, out)
        log.info("=== %s (%s)", name, path)
        res = run(path, PipelineParams(), log)
        results[name] = score(res["pred"], gt, VARIANTS[name][1])
        # per staff chain: how often its box is on the true staff member (centre within 50 px)
        inv = VARIANTS[name][1]
        t = res["tracks"].assign(chain=res["tracks"].track_id.map(res["chain"]))
        t = t[t.chain.isin(res["votes"].index[res["votes"].staff])]
        cx, cy = inv((t.x1 + t.x2).to_numpy() / 2, (t.y1 + t.y2).to_numpy() / 2)
        t = t.assign(cx=cx, cy=cy).merge(gt[["frame", "x", "y", "certain"]], on="frame")
        t["on_staff"] = np.hypot(t.cx - t.x, t.cy - t.y) <= 50
        for c, g in t.groupby("chain"):
            v = res["votes"].loc[c]
            members = sorted(k for k, cc in res["chain"].items() if cc == c)
            log.info("%s chain %d: hits %d, frames %d-%d, on staff %.0f%%, tracks %s",
                     name, c, v.hits, g.frame.min(), g.frame.max(), 100 * g.on_staff.mean(), members)
        # cache for fast re-voting experiments (scripts/step8_revote.py)
        res["tracks"].to_csv(out / f"{name}_tracks.csv", index=False)
        res["scores"].to_csv(out / f"{name}_tagscores.csv", index=False)
        results[name]["staff_chains"] = int(res["votes"].staff.sum()) if len(res["votes"]) else 0
        results[name]["seconds"] = round(res["seconds"], 1)
        log.info("%s: %s", name, results[name])
        (out / "results.json").write_text(json.dumps(results, indent=2))
    log.info("summary:\n%s", pd.DataFrame(results).T[["precision", "recall", "f1", "within_25px",
                                                      "median_err_px", "staff_chains", "seconds"]])


if __name__ == "__main__":
    main()
