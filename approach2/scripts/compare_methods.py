"""Compare view methods for person detection: run YOLO on each method's views,
map boxes back to the original frame, merge duplicates (NMS), and measure.

Per method and frame we record:
  n_boxes            people found after merging
  n_centre/middle/edge   the same, split by distance of the box centre from the image centre
  nested_pairs       pairs where one box's centre lies inside another box
                     (a hint of duplicates that NMS did not merge, or people overlapping)
  staff_hit          does any box contain the staff position from the answer key?
                     (only frames where the answer key is certain; NOTE the key was built
                     from approach 1's rotation boxes, so it favours the rotations method)
  views_ms, yolo_ms, post_ms   time to make views, run YOLO, map back + merge

Each method runs YOLO at its views' own size (no enlarging) unless its scale says otherwise,
so the comparison is "same pixel detail, different geometry".

Outputs:
  outputs/compare_methods/per_frame.csv, summary.csv, summary.md
  docs/figures/compare_frame<N>.jpg   merged boxes of every method on the original frame

Usage (from approach2/):
    ../.venv/bin/python scripts/compare_methods.py                 # 50 frames spread over the video
    ../.venv/bin/python scripts/compare_methods.py --n-frames 250 --start 600 --contiguous   # a 10 s clip
    ../.venv/bin/python scripts/compare_methods.py --device cpu
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from staffid2.common import MODELS_DIR, PROJECT_ROOT, ROOT, SAMPLE_VIDEO, setup_logging  # noqa: E402
from staffid2.detect import detect, pick_device, yolo_size  # noqa: E402
from staffid2.drawing import RED, draw_box, id_colour, label, mosaic  # noqa: E402
from staffid2.fisheye_view import (ViewMaker, cylindrical_maker, lens_focal, native_view_size, plain_maker,  # noqa: E402
                            quadrants_maker, rectified_maker, rotations_maker)

GT_CSV = PROJECT_ROOT / "approach1_groundtruth" / "ground_truth_draft.csv"


def build_methods(w: int, h: int, a: argparse.Namespace) -> dict[str, tuple[ViewMaker, float]]:
    """name -> (view maker, YOLO scale relative to the views' own size)."""
    c = (w / 2, h / 2)
    radius = float(np.hypot(w, h) / 2)
    f_fish = lens_focal(radius, a.fov_deg)
    rect_size = native_view_size(f_fish, a.view_fov_deg)
    return {
        "plain": (plain_maker(w, h), 1.0),
        "rotations": (rotations_maker(w, h), 1.0),
        "quad_axis": (quadrants_maker(w, h, c, a.quad_overlap, (0, 0, 180, 180)), 1.0),
        "quad_diag": (quadrants_maker(w, h, c, a.quad_overlap, (-45, 45, -135, 135)), 1.0),
        "quad_axis_2x": (quadrants_maker(w, h, c, a.quad_overlap, (0, 0, 180, 180)), 2.0),
        "rectified": (rectified_maker(w, h, c, f_fish, a.n_views, a.tilt_deg, a.view_fov_deg, rect_size), 1.0),
        "cylindrical": (cylindrical_maker(w, h, c, radius, int(2 * np.pi * radius), a.cyl_tiles, a.cyl_overlap), 1.0),
    }


def nested_pairs(boxes: np.ndarray) -> int:
    if len(boxes) < 2:
        return 0
    cx, cy = (boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2
    inside = ((cx[:, None] >= boxes[None, :, 0]) & (cx[:, None] <= boxes[None, :, 2]) &
              (cy[:, None] >= boxes[None, :, 1]) & (cy[:, None] <= boxes[None, :, 3]))
    np.fill_diagonal(inside, False)
    return int((inside | inside.T)[np.triu_indices(len(boxes), 1)].sum())


def ring_counts(boxes: np.ndarray, w: int, h: int, centre_r: float, edge_r: float) -> tuple[int, int, int]:
    d = np.hypot((boxes[:, 0] + boxes[:, 2]) / 2 - w / 2, (boxes[:, 1] + boxes[:, 3]) / 2 - h / 2)
    return int((d < centre_r).sum()), int(((d >= centre_r) & (d < edge_r)).sum()), int((d >= edge_r).sum())


def to_markdown(df: pd.DataFrame) -> str:
    """Plain markdown table (pandas' to_markdown needs the optional 'tabulate' package)."""
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def draw(frame: np.ndarray, boxes: np.ndarray, text: str, color: tuple, staff_xy: tuple | None) -> np.ndarray:
    out = frame.copy()
    for b in boxes:
        draw_box(out, b[:4], color)
    if staff_xy is not None:
        cv2.drawMarker(out, (int(staff_xy[0]), int(staff_xy[1])), RED, cv2.MARKER_CROSS, 24, 3)
    return label(out, text)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", default=str(SAMPLE_VIDEO))
    ap.add_argument("--weights", default=str(MODELS_DIR / "yolo11s.pt"))
    ap.add_argument("--gt", default=str(GT_CSV), help="answer key for staff_hit ('' to skip)")
    ap.add_argument("--n-frames", type=int, default=50)
    ap.add_argument("--contiguous", action="store_true", help="take n-frames in a row from --start instead of spreading")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--show-frames", type=int, nargs="*", default=[700, 1000], help="frames to draw for the report")
    ap.add_argument("--methods", nargs="*", default=None, help="subset of methods (default: all)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--nms-iou", type=float, default=0.45, help="same as approach 1")
    ap.add_argument("--fov-deg", type=float, default=180.0)
    ap.add_argument("--n-views", type=int, default=4)
    ap.add_argument("--tilt-deg", type=float, default=45.0)
    ap.add_argument("--view-fov-deg", type=float, default=90.0)
    ap.add_argument("--quad-overlap", type=float, default=0.15)
    ap.add_argument("--cyl-tiles", type=int, default=4)
    ap.add_argument("--cyl-overlap", type=float, default=0.2)
    ap.add_argument("--centre-r", type=float, default=150.0, help="ring boundary (px from image centre)")
    ap.add_argument("--edge-r", type=float, default=350.0, help="ring boundary (px from image centre)")
    ap.add_argument("--device", default="auto", help="auto | cpu | mps")
    ap.add_argument("--warmup", type=int, default=2, help="untimed frames per method before measuring")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "compare_methods"))
    ap.add_argument("--fig-dir", default=str(ROOT / "docs" / "figures"))
    args = ap.parse_args()

    log = setup_logging("compare_methods")
    device = pick_device(args.device)
    log.info("params: %s  device=%s", vars(args), device)
    out, fig_dir = Path(args.out), Path(args.fig_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ---- frames
    cap = cv2.VideoCapture(args.video)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.contiguous:
        idxs = list(range(args.start, min(args.start + args.n_frames, n_total)))
    else:
        idxs = sorted(set(np.linspace(0, n_total - 1, args.n_frames).astype(int).tolist()))
    wanted = sorted(set(idxs) | set(args.show_frames))
    frames: dict[int, np.ndarray] = {}
    for i in wanted:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = cap.read()
        if ok:
            frames[i] = f
    cap.release()
    h, w = next(iter(frames.values())).shape[:2]
    log.info("video: %s (%d frames, %dx%d); measuring %d frames: %s", args.video, n_total, w, h, len(idxs), idxs)

    gt = None
    if args.gt:
        g = pd.read_csv(args.gt).set_index("frame")
        gt = g[(g.staff_present == 1) & (g.pos_certain == 1)][["x", "y"]]
        log.info("answer key: %s (%d certain staff positions)", args.gt, len(gt))

    model = YOLO(args.weights)
    methods = build_methods(w, h, args)
    if args.methods:
        methods = {k: v for k, v in methods.items() if k in args.methods}

    rows, drawn = [], {i: [] for i in args.show_frames if i in frames}
    for mi, (name, (maker, scale)) in enumerate(methods.items()):
        sample_views = maker(frames[idxs[0]])
        imgsz = yolo_size(sample_views, scale)
        view_px = sum(v.img.shape[0] * v.img.shape[1] for v in sample_views) * scale ** 2
        log.info("method %-13s views=%d  view sizes=%s  scale=%.1f  imgsz=%d  view pixels=%.2f MP",
                 name, len(sample_views), sorted({v.img.shape[:2] for v in sample_views}), scale, imgsz, view_px / 1e6)
        for i in idxs[:args.warmup]:
            detect(model, maker(frames[i]), imgsz, args.conf, args.nms_iou, device)

        for i in idxs:
            t0 = time.perf_counter()
            views = maker(frames[i])
            views_ms = (time.perf_counter() - t0) * 1e3
            boxes, yolo_ms, post_ms = detect(model, views, imgsz, args.conf, args.nms_iou, device)
            c, m, e = ring_counts(boxes, w, h, args.centre_r, args.edge_r)
            hit = np.nan
            if gt is not None and i in gt.index:
                x, y = gt.loc[i, ["x", "y"]]
                hit = float(((boxes[:, 0] <= x) & (x <= boxes[:, 2]) & (boxes[:, 1] <= y) & (y <= boxes[:, 3])).any())
            rows.append(dict(method=name, frame=i, n_boxes=len(boxes), n_centre=c, n_middle=m, n_edge=e,
                             nested_pairs=nested_pairs(boxes), staff_hit=hit, imgsz=imgsz, n_views=len(views),
                             view_mp=view_px / 1e6, views_ms=views_ms, yolo_ms=yolo_ms, post_ms=post_ms))

        for i in drawn:
            boxes, *_ = detect(model, maker(frames[i]), imgsz, args.conf, args.nms_iou, device)
            staff = tuple(gt.loc[i, ["x", "y"]]) if gt is not None and i in gt.index else None
            drawn[i].append(draw(frames[i], boxes, f"{name}: {len(boxes)}", id_colour(mi), staff))
        log.info("  done: mean boxes=%.1f  mean yolo=%.1f ms",
                 np.mean([r["n_boxes"] for r in rows if r["method"] == name]),
                 np.mean([r["yolo_ms"] for r in rows if r["method"] == name]))

    df = pd.DataFrame(rows)
    df.to_csv(out / "per_frame.csv", index=False)
    df["total_ms"] = df.views_ms + df.yolo_ms + df.post_ms
    summary = df.groupby("method", sort=False).agg(
        n_views=("n_views", "first"), imgsz=("imgsz", "first"), view_mp=("view_mp", "first"),
        boxes_mean=("n_boxes", "mean"), boxes_median=("n_boxes", "median"),
        centre=("n_centre", "mean"), middle=("n_middle", "mean"), edge=("n_edge", "mean"),
        nested_pairs=("nested_pairs", "mean"), staff_hit_rate=("staff_hit", "mean"), staff_frames=("staff_hit", "count"),
        views_ms=("views_ms", "mean"), yolo_ms=("yolo_ms", "mean"), post_ms=("post_ms", "mean"), total_ms=("total_ms", "mean"),
    ).round(2)
    summary["fps"] = (1000 / summary.total_ms).round(1)
    summary.to_csv(out / "summary.csv")
    (out / "summary.md").write_text(to_markdown(summary.reset_index()))
    log.info("summary (device=%s, %d frames):\n%s", device, len(idxs), summary.to_string())

    for i, panels in drawn.items():
        small = [cv2.resize(p, (w // 2, h // 2)) for p in [label(frames[i], f"original frame {i}")] + panels]
        path = fig_dir / f"compare_frame{i}.jpg"
        cv2.imwrite(str(path), mosaic(small, cols=4))
        log.info("figure: %s", path)
    log.info("outputs: %s", out)


if __name__ == "__main__":
    main()
