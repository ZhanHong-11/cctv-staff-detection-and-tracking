"""Plain YOLO11s person detection on every frame, drawn onto an output video.

No rotations, no tracking: this shows what an off-the-shelf COCO model sees
(and misses) in the overhead fisheye view.

Usage (from approach2/):
    ../.venv/bin/python scripts/yolo11_demo.py
    ../.venv/bin/python scripts/yolo11_demo.py --video <file> --conf 0.25 --device mps
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
from staffid2.detect import PERSON_CLASS, pick_device  # noqa: E402
from staffid2.drawing import GREEN, draw_box, draw_header  # noqa: E402


def draw(frame: np.ndarray, boxes: np.ndarray, confs: np.ndarray, idx: int) -> np.ndarray:
    for box, c in zip(boxes, confs):
        draw_box(frame, box, GREEN, f"{c:.2f}")
    draw_header(frame, f"frame {idx}  persons {len(boxes)}")
    return frame


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", default=str(SAMPLE_VIDEO))
    ap.add_argument("--weights", default=str(MODELS_DIR / "yolo11s.pt"))
    ap.add_argument("--conf", type=float, default=0.25, help="minimum confidence to keep a box")
    ap.add_argument("--imgsz", type=int, default=960, help="YOLO input size (video is 960 wide)")
    ap.add_argument("--device", default="auto", help="auto | cpu | mps")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "yolo11_demo.mp4"))
    args = ap.parse_args()

    log = setup_logging("yolo11_demo")
    device = pick_device(args.device)
    log.info("params: %s  device=%s", vars(args), device)

    model = YOLO(args.weights)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    log.info("video: %s  %dx%d  %.1f fps  %d frames", args.video, w, h, fps, n_frames)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = open_writer(out, fps, (w, h))

    counts: list[int] = []
    t0 = time.time()
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = model.predict(frame, classes=[PERSON_CLASS], conf=args.conf, imgsz=args.imgsz,
                            device=device, verbose=False)[0]
        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        counts.append(len(boxes))
        writer.write(draw(frame, boxes, confs, idx))
        if idx % 100 == 0:
            log.info("frame %d/%d  persons=%d", idx, n_frames, len(boxes))
        idx += 1
    cap.release()
    writer.release()

    elapsed = time.time() - t0
    c = np.array(counts)
    log.info("done: %d frames in %.1f s (%.1f fps)", idx, elapsed, idx / max(elapsed, 1e-9))
    log.info("persons per frame: min=%d  median=%.0f  mean=%.1f  max=%d", c.min(), np.median(c), c.mean(), c.max())
    log.info("output video: %s", out)


if __name__ == "__main__":
    main()
