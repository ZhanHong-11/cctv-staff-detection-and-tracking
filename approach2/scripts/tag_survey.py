"""Step 3a, part 3: measure the tags the user labelled (ground_truth/tag_boxes.csv) and show
how much variety the sample video really contains.

For every box: box size and squash (h/w), the tag's own length, thickness and rotation, camera tilt and position in the image, walking speed,
shirt brightness around the tag, tag-vs-shirt contrast, bar contrast, number of bars resolved
in the best single pixel row, edge sharpness and rotation of the tag's long axis
(see src/staffid2/tag_measure.py).
Shirt = "dark" when the median brightness around the tag is below --dark-max, else "light".

Writes
- outputs/tag_survey/tag_measurements.csv  one row per box
- outputs/tag_survey/sightings.csv         one row per sighting (ranges and medians)
- outputs/tag_survey/ranges.csv            min / 5% / median / 95% / max per measurement and shirt
- docs/figures/step3a_gallery.png          every labelled tag, enlarged, one row per sighting
- docs/figures/step3a_size_vs_tilt.png     width and squash vs tilt, rotation, sharpness vs speed
- docs/figures/step3a_contrast.png         tag-vs-shirt and bar contrast per shirt
- docs/figures/step3a_profiles.png         brightness along the best row, one per sighting
- docs/figures/step3a_coverage.png         where in the image the labelled tags are

Usage (from approach2/):
    ../.venv/bin/python scripts/tag_survey.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from staffid2.common import ROOT, SAMPLE_VIDEO, setup_logging  # noqa: E402
from staffid2.person_crop import Lens, person_view  # noqa: E402
from staffid2.tag_measure import best_row, measure  # noqa: E402

SHIRT_STYLE = {"light": dict(color="#2a78d6", marker="o", label="light shirt"),
               "dark": dict(color="#eb6834", marker="s", label="dark hoodie")}
MEASURES = ["w", "h", "squash", "length", "thickness", "shape_ratio", "tilt_deg", "speed_px", "shirt_median", "tag_vs_shirt", "bar_contrast",
            "n_dips", "sharpness", "angle_deg"]


def read_frames(video: str, frames: list[int]) -> dict[int, np.ndarray]:
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video}")
    out, i, want = {}, -1, set(frames)
    while want:
        ok, img = cap.read()
        i += 1
        if not ok:
            break
        if i in want:
            out[i] = img
            want.discard(i)
    cap.release()
    return out


def speed(key: pd.DataFrame, f: int) -> float:
    """Staff box-centre speed (px/frame) from the answer key, central difference."""
    a, b = key.loc[max(f - 1, key.index.min())], key.loc[min(f + 1, key.index.max())]
    span = max(int(b.name) - int(a.name), 1)
    return float(np.hypot(b.x - a.x, b.y - a.y) / span)


def gallery(rows: pd.DataFrame, crops: dict[int, np.ndarray], cell: int, pad: int) -> np.ndarray:
    """One line per sighting: each tag with `pad` px around it, scaled (nearest) to cell height."""
    lines = []
    for s, g in rows.groupby("sighting"):
        tiles = []
        for r in g.itertuples():
            img = crops[r.frame]
            x1, y1 = max(r.x1 - pad, 0), max(r.y1 - pad, 0)
            patch = img[y1:r.y2 + pad, x1:r.x2 + pad]
            k = cell / patch.shape[0]
            big = cv2.resize(patch, (max(1, round(patch.shape[1] * k)), cell), interpolation=cv2.INTER_NEAREST)
            cv2.rectangle(big, (round((r.x1 - x1) * k), round((r.y1 - y1) * k)),
                          (round((r.x2 - x1) * k) - 1, round((r.y2 - y1) * k) - 1), (0, 0, 255), 1)
            cap = np.full((16, big.shape[1], 3), 255, np.uint8)
            cv2.putText(cap, f"{r.frame} {r.w}x{r.h}", (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (40, 40, 40), 1)
            tiles.append(np.vstack([cap, big]))
            tiles.append(np.full((cell + 16, 6, 3), 255, np.uint8))
        head = np.full((cell + 16, 120, 3), 255, np.uint8)
        cv2.putText(head, f"sighting {s}", (4, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1)
        cv2.putText(head, f"{g.frame.min()}-{g.frame.max()}", (4, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 90, 90), 1)
        cv2.putText(head, g.shirt.iloc[0], (4, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 90, 90), 1)
        lines.append(np.hstack([head] + tiles))
        lines.append(np.full((8, 1, 3), 255, np.uint8))
    width = max(l.shape[1] for l in lines)
    return np.vstack([cv2.copyMakeBorder(l, 0, 0, 0, width - l.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
                      for l in lines])


def style_axes(ax) -> None:
    ax.grid(True, color="#e4e4e0", lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#9a9a94")
    ax.tick_params(colors="#55554f", labelsize=8)


def scatter_by_shirt(ax, t: pd.DataFrame, x: str, y: str) -> None:
    for shirt, st in SHIRT_STYLE.items():
        g = t[t.shirt == shirt]
        ax.scatter(g[x], g[y], s=28, c=st["color"], marker=st["marker"], label=st["label"],
                   edgecolors="white", linewidths=0.8)
    style_axes(ax)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", default=str(SAMPLE_VIDEO))
    ap.add_argument("--answer-key", default=str(ROOT / "ground_truth" / "ground_truth.csv"))
    ap.add_argument("--tag-boxes", default=str(ROOT / "ground_truth" / "tag_boxes.csv"))
    ap.add_argument("--tag-sightings", default=str(ROOT / "ground_truth" / "tag_sightings.csv"))
    ap.add_argument("--fov-deg", type=float, default=180.0, help="lens field of view across 2*radius")
    ap.add_argument("--ring", type=int, default=3, help="width of the shirt band around the tag (px)")
    ap.add_argument("--dark-max", type=float, default=100.0, help="shirt brightness below this = dark")
    ap.add_argument("--dip-frac", type=float, default=0.15, help="bar dip: prominence >= this x profile range")
    ap.add_argument("--dip-abs", type=float, default=8.0, help="bar dip: prominence >= this many gray levels")
    ap.add_argument("--gallery-cell", type=int, default=72, help="gallery tile height (px)")
    ap.add_argument("--gallery-pad", type=int, default=4, help="crop px shown around each tag in the gallery")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "tag_survey"))
    ap.add_argument("--fig-dir", default=str(ROOT / "docs" / "figures"))
    args = ap.parse_args()

    log = setup_logging("tag_survey")
    log.info("params: %s", vars(args))
    t0 = time.time()
    out, figs = Path(args.out), Path(args.fig_dir)
    out.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    key = pd.read_csv(args.answer_key).set_index("frame")
    boxes = pd.read_csv(args.tag_boxes)
    sights = pd.read_csv(args.tag_sightings)
    frames = read_frames(args.video, boxes.frame.tolist())
    h, w = next(iter(frames.values())).shape[:2]
    lens = Lens.for_frame(w, h, args.fov_deg)
    log.info("%d boxes in %d sightings, %d frames read", len(boxes), len(sights), len(frames))

    rows, crops = [], {}
    for r in boxes.itertuples():
        view, p = person_view(frames[r.frame], (r.staff_x1, r.staff_y1, r.staff_x2, r.staff_y2), lens, r.margin)
        if p["size"] != r.crop_size:
            log.warning("frame %d: crop size %d != labelled %d (answer key changed?)", r.frame, p["size"], r.crop_size)
        crops[r.frame] = view.img
        gray = cv2.cvtColor(view.img, cv2.COLOR_BGR2GRAY)
        box = (int(r.x1), int(r.y1), int(r.x2), int(r.y2))
        m = measure(gray, box, args.ring, args.dip_frac, args.dip_abs)
        rows.append(dict(frame=r.frame, sighting=r.sighting, x1=r.x1, y1=r.y1, x2=r.x2, y2=r.y2, w=r.w, h=r.h,
                         squash=r.h / r.w, tilt_deg=r.tilt_deg, azimuth_deg=r.azimuth_deg,
                         radius_px=float(np.hypot(r.orig_x - w / 2, r.orig_y - h / 2)),
                         orig_x=r.orig_x, orig_y=r.orig_y, speed_px=speed(key, r.frame),
                         staff_source=r.staff_source, **m))
    t = pd.DataFrame(rows)
    t["shape_ratio"] = t.thickness / t.length   # squash of the tag itself, rotation removed
    t["shirt"] = np.where(t.shirt_median < args.dark_max, "dark", "light")
    t.round(3).to_csv(out / "tag_measurements.csv", index=False)

    # one row per sighting
    srows = []
    for s in sights.itertuples():
        g = t[t.sighting == s.sighting]
        srows.append(dict(sighting=s.sighting, start=s.start, end=s.end, n_frames=s.n_frames, n_boxes=len(g),
                          shirt=g.shirt.mode().iloc[0] if len(g) else "", w_min=g.w.min(), w_max=g.w.max(),
                          h_min=g.h.min(), h_max=g.h.max(), tilt_min=g.tilt_deg.min(), tilt_max=g.tilt_deg.max(),
                          orig_x=g.orig_x.median(), orig_y=g.orig_y.median(), speed_med=g.speed_px.median(),
                          dips_med=g.n_dips.median(), sharpness_med=g.sharpness.median()))
    st = pd.DataFrame(srows)
    st.round(2).to_csv(out / "sightings.csv", index=False)
    log.info("sightings:\n%s", st.round(1).to_string(index=False))

    # ranges per measurement, overall and per shirt
    rr = []
    for group, g in [("all", t)] + list(t.groupby("shirt")):
        for m in MEASURES:
            q = np.percentile(g[m], [0, 5, 50, 95, 100])
            rr.append(dict(group=group, measure=m, n=len(g), min=q[0], p5=q[1], median=q[2], p95=q[3], max=q[4]))
    ranges = pd.DataFrame(rr)
    ranges.round(2).to_csv(out / "ranges.csv", index=False)
    log.info("ranges:\n%s", ranges.round(2).to_string(index=False))
    log.info("bars resolved (>= 2 dips): %d of %d boxes; 3 dips: %d", (t.n_dips >= 2).sum(), len(t), (t.n_dips == 3).sum())

    # ---------------------------------------------------------------- figures
    cv2.imwrite(str(figs / "step3a_gallery.png"), gallery(t, crops, args.gallery_cell, args.gallery_pad))

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4))
    axes = axes.ravel()
    scatter_by_shirt(axes[0], t, "tilt_deg", "length")
    axes[0].set(xlabel="camera tilt (deg, 0 = straight down)", ylabel="tag length along its axis (px)")
    axes[0].set_title("Tag gets smaller away from the centre", fontsize=10, loc="left")
    scatter_by_shirt(axes[1], t, "tilt_deg", "shape_ratio")
    axes[1].set(xlabel="camera tilt (deg)", ylabel="thickness / length (rotation removed)")
    axes[1].set_title("Tag squash vs tilt", fontsize=10, loc="left")
    scatter_by_shirt(axes[2], t, "tilt_deg", "angle_deg")
    axes[2].set(xlabel="camera tilt (deg)", ylabel="long-axis rotation (deg, + = clockwise)")
    axes[2].set_title("Tag rotation in the rectified crop", fontsize=10, loc="left")
    axes[2].axhline(0, color="#9a9a94", lw=1, ls="--")
    scatter_by_shirt(axes[3], t, "speed_px", "sharpness")
    axes[3].set(xlabel="walking speed (px / frame)", ylabel="edge sharpness (1 = crisp)")
    axes[3].set_title("Sharpness vs walking speed", fontsize=10, loc="left")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figs / "step3a_size_vs_tilt.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    rng = np.random.default_rng(0)
    for ax, m, lab in zip(axes, ["shirt_median", "tag_vs_shirt", "bar_contrast"],
                          ["shirt brightness around the tag", "tag white - shirt (gray levels)",
                           "bar contrast inside the tag (gray levels)"]):
        for k, (shirt, sty) in enumerate(SHIRT_STYLE.items()):
            g = t[t.shirt == shirt]
            ax.scatter(k + rng.uniform(-0.12, 0.12, len(g)), g[m], s=24, c=sty["color"], marker=sty["marker"],
                       edgecolors="white", linewidths=0.8)
            ax.plot([k - 0.25, k + 0.25], [g[m].median()] * 2, color="#33332f", lw=2)
        ax.set_xticks([0, 1], [s["label"] for s in SHIRT_STYLE.values()])
        ax.set_title(lab, fontsize=10, loc="left")
        style_axes(ax)
    axes[1].axhline(0, color="#9a9a94", lw=1, ls="--")
    fig.tight_layout()
    fig.savefig(figs / "step3a_contrast.png", dpi=130)
    plt.close(fig)

    # one profile per sighting: the box closest to the sighting's median width
    n = len(st)
    cols = 4
    fig, axes = plt.subplots(int(np.ceil(n / cols)), cols, figsize=(12, 2.4 * np.ceil(n / cols)), squeeze=False)
    for ax, s in zip(axes.ravel(), st.itertuples()):
        g = t[t.sighting == s.sighting]
        r = g.iloc[(g.w - g.w.median()).abs().argmin()]
        gray = cv2.cvtColor(crops[int(r.frame)], cv2.COLOR_BGR2GRAY)
        y, prof, _ = best_row(gray, (int(r.x1), int(r.y1), int(r.x2), int(r.y2)), args.dip_frac, args.dip_abs)
        sty = SHIRT_STYLE[r.shirt]
        ax.plot(np.arange(len(prof)), prof, "-", color=sty["color"], lw=2, marker=sty["marker"], ms=4)
        ax.set_title(f"sighting {s.sighting}: frame {int(r.frame)}, {int(r.w)}x{int(r.h)} px, row {y - int(r.y1)}, {int(r.n_dips)} dips",
                     fontsize=7.5, loc="left")
        ax.set_ylim(0, 255)
        style_axes(ax)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.supxlabel("column across the tag (px, 1 px margin each side)", fontsize=9)
    fig.supylabel("gray level", fontsize=9)
    fig.tight_layout()
    fig.savefig(figs / "step3a_profiles.png", dpi=130)
    plt.close(fig)

    bg = read_frames(args.video, [int(t.frame.iloc[0])])[int(t.frame.iloc[0])]
    bg = cv2.cvtColor(cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2RGB)
    fig, ax = plt.subplots(figsize=(8, 6.2))
    ax.imshow((bg * 0.45 + 255 * 0.55).astype(np.uint8))
    path = key[key.staff_present == 1]
    ax.plot(path.x, path.y, color="#9a9a94", lw=0.8, label="staff path (answer key)")
    for shirt, sty in SHIRT_STYLE.items():
        g = t[t.shirt == shirt]
        ax.scatter(g.orig_x, g.orig_y, s=30, c=sty["color"], marker=sty["marker"], label=f"labelled tag, {sty['label']}",
                   edgecolors="white", linewidths=0.8, zorder=3)
    for s in st.itertuples():
        ax.annotate(str(s.sighting), (s.orig_x, s.orig_y), xytext=(6, -10), textcoords="offset points",
                    fontsize=8, color="#33332f")
    ax.plot(w / 2, h / 2, "+", color="#33332f", ms=12)
    ax.set_axis_off()
    ax.legend(loc="lower left", fontsize=8, frameon=True)
    ax.set_title("Where the labelled tags are (numbers = sighting)", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(figs / "step3a_coverage.png", dpi=130)
    plt.close(fig)

    log.info("wrote %s and 5 figures in %s (%.1f s)", out, figs, time.time() - t0)


if __name__ == "__main__":
    main()
