"""Step 1: figures used to explore the video (saved to docs/figures/).

- step1_contact_sheet.jpg: every 56th frame, to see the scene and who moves
- step1_tag_zoom.jpg: staff crops in both outfits (white shirt, black hoodie)
- step1_tag_pixels.jpg: the tag at 10x with a 5 px grid, to measure its size
- step1_exposure_dip.jpg: frames around the brightness drop at ~1300
- step1_frame_diff.png: mean absolute difference between consecutive frames

The frame numbers and positions below were found by hand while exploring this
video; they are for illustration only and are not used by the pipeline.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from staffid.common import ROOT, SAMPLE_VIDEO, iter_frames, setup_logging  # noqa: E402

VIDEO = SAMPLE_VIDEO
FIG = ROOT / "docs" / "figures"

# frame -> person box (x, y, w, h) from the step 1 motion boxes
PERSON_BOXES = {380: (477, 0, 89, 172), 470: (486, 73, 82, 180), 1020: (496, 28, 85, 176)}
# frame -> approximate tag centre (x, y), measured by eye
TAG_CENTRES = {380: (532, 57), 470: (529, 126), 1020: (553, 92)}
DIP_FRAMES = [1298, 1300, 1302, 1306, 1340]


def label(img: np.ndarray, text: str) -> np.ndarray:
    return cv2.putText(img, text, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)


def main() -> None:
    log = setup_logging("step1_figures")
    FIG.mkdir(parents=True, exist_ok=True)
    contact, persons, tags, dip, diffs = [], [], [], [], []
    prev = None
    for i, f in iter_frames(VIDEO):
        g = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (160, 120)).astype(np.float32)
        if prev is not None:
            diffs.append(float(np.abs(g - prev).mean()))
        prev = g
        if i % 56 == 0 and len(contact) < 24:
            contact.append(label(cv2.resize(f, (320, 240)), str(i)))
        if i in PERSON_BOXES:
            x, y, w, h = PERSON_BOXES[i]
            persons.append(label(cv2.resize(f[y:y + h, x:x + w], (400, 720), interpolation=cv2.INTER_CUBIC), str(i)))
        if i in TAG_CENTRES:
            cx, cy = TAG_CENTRES[i]
            c = cv2.resize(f[cy - 20:cy + 20, cx - 30:cx + 30], None, fx=10, fy=10, interpolation=cv2.INTER_NEAREST)
            for k in range(0, c.shape[1], 50):
                cv2.line(c, (k, 0), (k, c.shape[0] - 1), (0, 255, 0), 1)
            for k in range(0, c.shape[0], 50):
                cv2.line(c, (0, k), (c.shape[1] - 1, k), (0, 255, 0), 1)
            tags.append(label(c, str(i)))
        if i in DIP_FRAMES:
            dip.append(label(cv2.resize(f, (384, 288)), str(i)))

    cv2.imwrite(str(FIG / "step1_contact_sheet.jpg"), np.vstack([np.hstack(contact[r * 6:r * 6 + 6]) for r in range(4)]))
    cv2.imwrite(str(FIG / "step1_tag_zoom.jpg"), np.hstack(persons))
    cv2.imwrite(str(FIG / "step1_tag_pixels.jpg"), np.hstack(tags))
    cv2.imwrite(str(FIG / "step1_exposure_dip.jpg"), np.hstack(dip))

    # frame-difference plot drawn with OpenCV (matplotlib is not installed yet)
    h, w, top = 300, 1341, 30.0
    plot = np.full((h, w, 3), 255, np.uint8)
    for x, d in enumerate(diffs):
        cv2.line(plot, (x, h - 1), (x, h - 1 - int(min(d, top) / top * (h - 20))), (180, 80, 0), 1)
    label(plot, "mean |frame diff| (clipped at 30); x = frame index")
    cv2.imwrite(str(FIG / "step1_frame_diff.png"), plot)

    top5 = sorted(range(len(diffs)), key=lambda k: -diffs[k])[:5]
    log.info("largest frame differences: %s", [(k + 1, round(diffs[k], 1)) for k in top5])
    log.info("median frame difference: %.2f", float(np.median(diffs)))
    log.info("figures written to %s", FIG)


if __name__ == "__main__":
    main()
