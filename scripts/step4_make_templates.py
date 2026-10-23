"""Step 4A: cut tag templates from upright staff crops.

The tag positions below were marked by eye on upright crops (data/review/upright.jpg,
coordinates in the 360x540 upright crop = 3x upscale), then refined automatically:
on dark clothing the tag is the brightest blob near the mark, on white clothing it
is the darkest cluster. Templates are saved to models/tag_templates/ and a preview
sheet to docs/figures/step4_templates.png.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from staffid.common import ROOT, iter_frames, setup_logging  # noqa: E402
from staffid.tag import TEMPLATE_UPSCALE, UprightParams, upright_crop  # noqa: E402

# frame -> (x, y) of the tag in the upright crop, marked by eye; "light" = white shirt
MARKS = {
    380: ((200, 208), "light"), 470: ((200, 210), "light"),
    859: ((214, 274), "dark"), 928: ((216, 230), "dark"),
    936: ((230, 230), "dark"), 975: ((180, 220), "dark"), 1063: ((200, 220), "dark"),
    1105: ((156, 210), "dark"), 1199: ((220, 200), "dark"),
}
TPL_W, TPL_H = 54, 33   # template size in upscaled pixels (~18x11 original px)
SEARCH = 24             # refinement radius around the mark


def refine(gray: np.ndarray, xy: tuple[int, int], kind: str) -> tuple[int, int]:
    """Move the mark to the centroid of the nearest tag-coloured blob (Otsu threshold)."""
    x, y = xy
    win = cv2.GaussianBlur(gray[y - SEARCH:y + SEARCH, x - SEARCH:x + SEARCH], (5, 5), 0)
    # bright patch on dark clothing, dark bars on light clothing
    mode = cv2.THRESH_BINARY if kind == "dark" else cv2.THRESH_BINARY_INV
    _, bw = cv2.threshold(win, 0, 255, mode + cv2.THRESH_OTSU)
    if kind == "light":  # join the separate dark bars into one blob
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((5, 9), np.uint8))
    n, _, stats, cents = cv2.connectedComponentsWithStats(bw)
    if n <= 1:
        return x, y
    d = [np.hypot(*(cents[k] - SEARCH)) if stats[k, cv2.CC_STAT_AREA] >= 20 else 1e9 for k in range(1, n)]
    cx, cy = cents[1 + int(np.argmin(d))]
    return x - SEARCH + int(round(cx)), y - SEARCH + int(round(cy))


def main() -> None:
    log = setup_logging("step4_make_templates")
    gt = pd.read_csv(ROOT / "ground_truth" / "ground_truth_draft.csv")
    out = ROOT / "models" / "tag_templates"
    out.mkdir(parents=True, exist_ok=True)
    previews = []
    for i, frame in iter_frames(ROOT / "AI Test Candidates.mp4"):
        if i not in MARKS:
            continue
        r = gt.iloc[i]
        up, _ = upright_crop(frame, (r.x1, r.y1, r.x2, r.y2), UprightParams(upscale=TEMPLATE_UPSCALE))
        gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
        (mx, my), kind = MARKS[i]
        x, y = refine(gray, (mx, my), kind)
        tpl = gray[y - TPL_H // 2:y + TPL_H // 2 + 1, x - TPL_W // 2:x + TPL_W // 2 + 1]
        cv2.imwrite(str(out / f"tag_{i:04d}_{kind}.png"), tpl)
        log.info("frame %d (%s): mark %s -> refined (%d, %d)", i, kind, (mx, my), x, y)
        p = cv2.resize(tpl, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
        previews.append(cv2.putText(cv2.cvtColor(p, cv2.COLOR_GRAY2BGR), str(i), (2, 14),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1))
    sheet = np.hstack([cv2.copyMakeBorder(p, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(0, 255, 0)) for p in previews])
    (ROOT / "docs" / "figures").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(ROOT / "docs" / "figures" / "step4_templates.png"), sheet)
    log.info("wrote %d templates to %s", len(previews), out)


if __name__ == "__main__":
    main()
