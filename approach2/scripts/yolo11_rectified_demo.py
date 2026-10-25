"""YOLO11s on rectified views for every frame; boxes mapped back to the original
frame, duplicates merged, drawn onto an output video.

Same idea as yolo11_demo.py (plain YOLO), so the two videos can be compared side by side.
Each frame is turned into --n-views virtual perspective cameras (tilted --tilt-deg away
from straight down, --view-fov-deg wide, sized to keep the fisheye's detail), YOLO runs on
all views in one batch, every box is mapped back to the original frame and NMS merges
duplicates from overlapping views. Box colour = which view found it.

Lens model (assumption, not calibrated): equidistant fisheye, 180 deg across a circle
centred on the image with radius = half the diagonal. See src/staffid2/fisheye_view.py.

Usage (from approach2/):
    ../.venv/bin/python scripts/yolo11_rectified_demo.py
    ../.venv/bin/python scripts/yolo11_rectified_demo.py --video <file> --device cpu --show-views
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from staffid2.common import MODELS_DIR, ROOT, SAMPLE_VIDEO, open_writer, setup_logging  # noqa: E402
from staffid2.detect import detect, pick_device, yolo_size  # noqa: E402
from staffid2.drawing import draw_box, draw_header, id_colour, label, mosaic  # noqa: E402
from staffid2.fisheye_view import lens_focal, native_view_size, rectified_maker  # noqa: E402


def draw(frame: np.ndarray, boxes: np.ndarray, idx: int) -> np.ndarray:
    out = frame.copy()
    for x1, y1, x2, y2, c, k in boxes:
        draw_box(out, (x1, y1, x2, y2), id_colour(int(k)), f"{c:.2f}")
    draw_header(out, f"frame {idx}  persons {len(boxes)}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", default=str(SAMPLE_VIDEO))
    ap.add_argument("--weights", default=str(MODELS_DIR / "yolo11s.pt"))
    ap.add_argument("--conf", type=float, default=0.25, help="minimum confidence to keep a box")
    ap.add_argument("--nms-iou", type=float, default=0.45, help="merge boxes overlapping more than this")
    ap.add_argument("--fov-deg", type=float, default=180.0, help="lens field of view across 2*radius")
    ap.add_argument("--radius", type=float, default=None, help="fisheye radius in px (default: half diagonal)")
    ap.add_argument("--n-views", type=int, default=4)
    ap.add_argument("--tilt-deg", type=float, default=45.0, help="virtual camera tilt from straight down")
    ap.add_argument("--view-fov-deg", type=float, default=90.0)
    ap.add_argument("--view-size", type=int, default=None, help="default: native (same detail as the fisheye)")
    ap.add_argument("--start-deg", type=float, default=90.0, help="direction of the first view (90 = bottom)")
    ap.add_argument("--device", default="auto", help="auto | cpu | mps")
    ap.add_argument("--show-views", action="store_true", help="also write a video of the rectified views")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "yolo11_rectified_demo.mp4"))
    args = ap.parse_args()

    log = setup_logging("yolo11_rectified_demo")
    device = pick_device(args.device)
    log.info("params: %s  device=%s", vars(args), device)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    log.info("video: %s  %dx%d  %.1f fps  %d frames", args.video, w, h, fps, n_frames)

    radius = args.radius if args.radius is not None else float(np.hypot(w, h) / 2)
    f_fish = lens_focal(radius, args.fov_deg)
    size = args.view_size or native_view_size(f_fish, args.view_fov_deg)
    maker = rectified_maker(w, h, (w / 2, h / 2), f_fish, args.n_views, args.tilt_deg,
                            args.view_fov_deg, size, args.start_deg)
    log.info("rectified: %d views of %dx%d px, radius=%.0f f_fish=%.1f", args.n_views, size, size, radius, f_fish)

    model = YOLO(args.weights)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = open_writer(out, fps, (w, h))
    views_writer, views_out = None, out.with_name(out.stem + "_views.mp4")

    counts: list[int] = []
    t_views = t_yolo = t_post = 0.0
    imgsz = None
    t0 = time.time()
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = time.perf_counter()
        views = maker(frame)
        t_views += time.perf_counter() - t
        imgsz = imgsz or yolo_size(views)
        boxes, yolo_ms, post_ms = detect(model, views, imgsz, args.conf, args.nms_iou, device)
        t_yolo += yolo_ms / 1e3
        t_post += post_ms / 1e3
        counts.append(len(boxes))
        writer.write(draw(frame, boxes, idx))
        if args.show_views:
            grid = mosaic([label(v.img, v.name) for v in views], cols=2)
            if views_writer is None:
                views_writer = open_writer(views_out, fps, (grid.shape[1], grid.shape[0]))
            views_writer.write(grid)
        if idx % 100 == 0:
            log.info("frame %d/%d  persons=%d", idx, n_frames, len(boxes))
        idx += 1
    cap.release()
    writer.release()
    if views_writer is not None:
        views_writer.release()
        log.info("views video: %s", views_out)

    elapsed = time.time() - t0
    c = np.array(counts)
    log.info("done: %d frames in %.1f s (%.1f fps overall incl. reading/drawing/writing)",
             idx, elapsed, idx / max(elapsed, 1e-9))
    log.info("per frame: views %.1f ms, yolo %.1f ms (imgsz %d), map back + merge %.2f ms",
             t_views / idx * 1e3, t_yolo / idx * 1e3, imgsz, t_post / idx * 1e3)
    log.info("persons per frame: min=%d  median=%.0f  mean=%.1f  max=%d", c.min(), np.median(c), c.mean(), c.max())
    log.info("output video: %s", out)


if __name__ == "__main__":
    main()
