"""Four ways to reshape the overhead fisheye view, saved as pictures for sample frames.

1. rotations: the whole frame rotated 0/90/180/270 deg (approach 1). Every person
   is roughly upright in at least one copy.

2. quadrants: cut the frame into 4 quadrants around the centre (with some overlap
   so people on the cut lines are not lost) and rotate each by its own angle.
   No lens model needed: plain crop + rotation. Three variants (as in compare_methods.py):
     quad_axis     top as is, bottom rotated 180 deg
     quad_diag     each quadrant's outward diagonal turned to "up" (+-45 / +-135 deg)
     quad_axis_2x  quad_axis enlarged 2x (what YOLO gets in that variant)

3. rectified: several virtual perspective ("normal") cameras, each tilted away
   from straight down towards one direction, rotated so "outward" is up.

4. cylindrical: unroll the circle into a panorama strip. x = angle around the
   image centre, y = distance from the centre (top row = rim, bottom row = centre).

The geometry lives in src/staffid2/fisheye_view.py (shared with compare_methods.py).

Outputs (in outputs/fisheye_unwarp/): for each frame N
    frameN_original.jpg, frameN_rotations.jpg, frameN_quad_axis.jpg, frameN_quad_diag.jpg,
    frameN_quad_axis_2x.jpg, frameN_rectified.jpg, frameN_cylindrical.jpg

Usage (from approach2/):
    ../.venv/bin/python scripts/fisheye_unwarp.py
    ../.venv/bin/python scripts/fisheye_unwarp.py --frames 700 1000 --tilt-deg 50
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from staffid2.common import ROOT, SAMPLE_VIDEO, setup_logging  # noqa: E402
from staffid2.drawing import label, mosaic  # noqa: E402
from staffid2.fisheye_view import (cylindrical_maps, lens_focal, quadrants_maker,  # noqa: E402
                            rectified_maker, rotations_maker)


def read_frame(video: str, idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"cannot read frame {idx} from {video}")
    return frame


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", default=str(SAMPLE_VIDEO))
    ap.add_argument("--frames", type=int, nargs="+", default=[700, 1000])
    ap.add_argument("--cx", type=float, default=None, help="fisheye centre x (default: image centre)")
    ap.add_argument("--cy", type=float, default=None, help="fisheye centre y (default: image centre)")
    ap.add_argument("--radius", type=float, default=None, help="fisheye radius in px (default: half diagonal)")
    ap.add_argument("--fov-deg", type=float, default=180.0, help="lens field of view across 2*radius")
    ap.add_argument("--cyl-width", type=int, default=None, help="strip width (default: circumference 2*pi*radius)")
    ap.add_argument("--start-deg", type=float, default=90.0, help="where the strip/first view starts (90 = bottom)")
    ap.add_argument("--n-views", type=int, default=4)
    ap.add_argument("--tilt-deg", type=float, default=45.0, help="virtual camera tilt from straight down")
    ap.add_argument("--view-fov-deg", type=float, default=90.0)
    ap.add_argument("--view-size", type=int, default=480)
    ap.add_argument("--quad-overlap", type=float, default=0.15,
                    help="how far each quadrant reaches past the cut lines (fraction of quadrant size)")
    ap.add_argument("--quad-scale", type=float, default=2.0, help="enlargement for quad_axis_2x")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "fisheye_unwarp"))
    args = ap.parse_args()

    log = setup_logging("fisheye_unwarp")
    log.info("params: %s", vars(args))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for idx in args.frames:
        img = read_frame(args.video, idx)
        h, w = img.shape[:2]
        center = (args.cx if args.cx is not None else w / 2, args.cy if args.cy is not None else h / 2)
        radius = args.radius if args.radius is not None else float(np.hypot(w, h) / 2)
        f_fish = lens_focal(radius, args.fov_deg)
        cv2.imwrite(str(out / f"frame{idx}_original.jpg"), img)
        log.info("frame %d: center=(%.0f, %.0f) radius=%.0f f_fish=%.1f", idx, *center, radius, f_fish)

        makers = {
            "rotations": rotations_maker(w, h),
            "quad_axis": quadrants_maker(w, h, center, args.quad_overlap, (0, 0, 180, 180)),
            "quad_diag": quadrants_maker(w, h, center, args.quad_overlap, (-45, 45, -135, 135)),
            "quad_axis_2x": quadrants_maker(w, h, center, args.quad_overlap, (0, 0, 180, 180)),
            "rectified": rectified_maker(w, h, center, f_fish, args.n_views, args.tilt_deg,
                                         args.view_fov_deg, args.view_size, args.start_deg),
        }
        scales = {"quad_axis_2x": args.quad_scale}
        for m in ("rotations", "quad_axis", "quad_diag", "quad_axis_2x", "rectified", "cylindrical"):
            t0 = time.time()
            if m == "cylindrical":
                mx, my = cylindrical_maps(center, radius, args.cyl_width or int(2 * np.pi * radius), args.start_deg)
                res = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            else:
                s = scales.get(m, 1.0)
                views = makers[m](img)
                imgs = [cv2.resize(v.img, None, fx=s, fy=s) if s != 1.0 else v.img for v in views]
                res = mosaic([label(i, v.name) for i, v in zip(imgs, views)], cols=2)
            path = out / f"frame{idx}_{m}.jpg"
            cv2.imwrite(str(path), res)
            log.info("  %s: %dx%d in %.0f ms -> %s", m, res.shape[1], res.shape[0], (time.time() - t0) * 1e3, path)


if __name__ == "__main__":
    main()
