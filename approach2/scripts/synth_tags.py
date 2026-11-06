"""Step 3b: synthetic tags from the brief's reference photo, checked against the real tags.

1. Source: the green-shirt photo is taken out of the brief PDF, the tag is found automatically
   (tag_source.find_tag), turned into an albedo map, and its layout (band, 3 bars, outline) is
   measured to also draw a clean version. The brief's two video snapshots are NOT used.
2. Synthetic tags are rendered onto real torsos: chest area of other people's crops (negatives
   from staff_crops.py, never the staff member), placed where real tags sit in a crop.
3. Every synthetic tag is measured with the same functions as the real ones (step 3a), and the
   distributions are compared. The goal is that synthetic covers real, a bit wider.

Writes
- outputs/tag_source/photo_albedo.png, drawn_albedo.png, layout.json
- outputs/synth_tags/synth_measurements.csv, compare.csv
- docs/figures/step3b_source.png, step3b_gallery.png, step3b_compare.png

Usage (from approach2/):
    ../.venv/bin/python scripts/synth_tags.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from staffid2.common import BRIEF_PDF, ROOT, setup_logging  # noqa: E402
from staffid2.synth_tag import DEFAULT_RANGES, render, sample_params  # noqa: E402
from staffid2.tag_source import draw_tag, find_tag, measure_layout, pdf_jpegs, to_albedo  # noqa: E402
from staffid2.common import SAMPLE_VIDEO  # noqa: E402
from staffid2.person_crop import Lens, person_view  # noqa: E402
from staffid2.tag_measure import measure  # noqa: E402

REAL_COLOR, SYN_COLOR = "#2a78d6", "#eb6834"
COMPARE = [("length", "tag length (px)"), ("shape_ratio", "thickness / length"), ("angle_deg", "rotation (deg)"),
           ("tag_vs_shirt", "card white - shirt"), ("bar_contrast", "bar contrast"), ("sharpness", "edge sharpness"),
           ("n_dips", "bars resolved (dips in best row)")]


def look_for(shirt: float, dark_max: float, light_min: float) -> str:
    return "dark" if shirt < dark_max else ("light" if shirt >= light_min else "mid")


def source_figure(photo: np.ndarray, box, alb: np.ndarray, drawn: np.ndarray, path: Path) -> None:
    x1, y1, x2, y2 = box
    fig, ax = plt.subplots(1, 4, figsize=(13, 3.6), gridspec_kw={"width_ratios": [1.1, 1.2, 1, 1]})
    ax[0].imshow(cv2.cvtColor(photo, cv2.COLOR_BGR2RGB))
    ax[0].add_patch(plt.Rectangle((x1 - .5, y1 - .5), x2 - x1, y2 - y1, fill=False, ec="red", lw=1.5))
    ax[0].set_title(f"brief photo {photo.shape[1]}x{photo.shape[0]} px, tag found", fontsize=9, loc="left")
    pad = 6
    zoom = photo[max(y1 - pad, 0):y2 + pad, max(x1 - pad, 0):x2 + pad]
    ax[1].imshow(cv2.cvtColor(zoom, cv2.COLOR_BGR2RGB), interpolation="nearest")
    ax[1].add_patch(plt.Rectangle((pad - .5, pad - .5), x2 - x1, y2 - y1, fill=False, ec="red", lw=1.5))
    ax[1].set_title(f"tag {x2 - x1}x{y2 - y1} px", fontsize=9, loc="left")
    ax[2].imshow(alb, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax[2].set_title("photo albedo (0 ink, 1 white)", fontsize=9, loc="left")
    ax[3].imshow(drawn, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax[3].set_title("drawn tag (same layout)", fontsize=9, loc="left")
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def gallery(real: list[np.ndarray], syn: list[np.ndarray], cell: int, cols: int) -> np.ndarray:
    """Two blocks of enlarged patches: real tags (top) and synthetic tags (bottom)."""
    def block(patches: list[np.ndarray], title: str) -> np.ndarray:
        tiles = [cv2.resize(p, (cell, cell), interpolation=cv2.INTER_NEAREST) for p in patches]
        tiles += [np.full((cell, cell), 255, np.uint8)] * (-len(tiles) % cols)
        rows = [np.hstack([np.pad(t, 2, constant_values=255) for t in tiles[i:i + cols]])
                for i in range(0, len(tiles), cols)]
        head = np.full((26, rows[0].shape[1]), 255, np.uint8)
        cv2.putText(head, title, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, 30, 1)
        return np.vstack([head] + rows)
    a, b = block(real, "REAL tags (user labels, sample video)"), block(syn, "SYNTHETIC tags (from the brief photo / drawn)")
    return np.vstack([a, np.full((10, a.shape[1]), 255, np.uint8), b])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=600, help="synthetic tags to render and measure")
    ap.add_argument("--neg-crops", default=str(ROOT / "data" / "tag_crops"), help="staff_crops.py output")
    ap.add_argument("--real", default=str(ROOT / "outputs" / "tag_survey" / "tag_measurements.csv"))
    ap.add_argument("--tag-boxes", default=str(ROOT / "ground_truth" / "tag_boxes.csv"))
    ap.add_argument("--chest-x", type=float, nargs=2, default=(0.42, 0.62), help="tag centre range, fraction of crop")
    ap.add_argument("--chest-y", type=float, nargs=2, default=(0.25, 0.55))
    ap.add_argument("--dark-max", type=float, default=45.0, help="background level below this -> dark look")
    ap.add_argument("--light-min", type=float, default=110.0, help="background level from this -> light look")
    ap.add_argument("--max-tries", type=int, default=60, help="torsos tried to find the wanted look")
    ap.add_argument("--drawn-share", type=float, default=0.5, help="fraction rendered from the drawn tag")
    ap.add_argument("--ring", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gallery-n", type=int, default=40)
    ap.add_argument("--out", default=str(ROOT / "outputs"))
    ap.add_argument("--fig-dir", default=str(ROOT / "docs" / "figures"))
    args = ap.parse_args()

    log = setup_logging("synth_tags")
    log.info("params: %s", vars(args))
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    out, figs = Path(args.out), Path(args.fig_dir)
    (out / "tag_source").mkdir(parents=True, exist_ok=True)
    (out / "synth_tags").mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    # 1. source
    imgs = pdf_jpegs(BRIEF_PDF)
    log.info("brief PDF: %d JPEGs %s; using #0 (reference photo) only", len(imgs), [i.shape[:2] for i in imgs])
    photo = imgs[0]
    box = find_tag(photo)
    alb = to_albedo(photo, box)
    layout = measure_layout(alb)
    drawn = draw_tag(layout, height=120)
    log.info("tag box %s (%dx%d px), layout %s", box, box[2] - box[0], box[3] - box[1], layout)
    cv2.imwrite(str(out / "tag_source" / "photo_albedo.png"), (alb * 255).astype(np.uint8))
    cv2.imwrite(str(out / "tag_source" / "drawn_albedo.png"), (drawn * 255).astype(np.uint8))
    (out / "tag_source" / "layout.json").write_text(json.dumps(dict(box=box, **asdict(layout)), indent=1))
    source_figure(photo, box, alb, drawn, figs / "step3b_source.png")

    # 2. render on real torsos + 3. measure
    neg = pd.read_csv(Path(args.neg_crops) / "crops.csv")
    neg = neg[neg.role == "neg"].reset_index(drop=True)
    rows, patches = [], []
    looks, look_p = ("light", "dark", "mid"), np.asarray(DEFAULT_RANGES["look_p"])
    for i in range(args.n):
        # pick the look first, then a torso of matching brightness (most people in this clip wear
        # dark clothes, so drawing torsos at random would give few light-shirt tags)
        want = str(rng.choice(looks, p=look_p / look_p.sum()))
        for _ in range(args.max_tries):
            r = neg.iloc[int(rng.integers(len(neg)))]
            crop = cv2.imread(str(Path(args.neg_crops) / "neg" / f"f{int(r.frame):04d}_t{int(r.track_id)}.png"),
                              cv2.IMREAD_GRAYSCALE)
            size = crop.shape[0]
            cx, cy = rng.uniform(*args.chest_x) * size, rng.uniform(*args.chest_y) * size
            s_local = float(np.median(crop[max(int(cy) - 6, 0):int(cy) + 6, max(int(cx) - 6, 0):int(cx) + 6]))
            look = look_for(s_local, args.dark_max, args.light_min)
            if look == want:
                break
        p = sample_params(rng, shirt=s_local, look=look)
        source = "drawn" if rng.random() < args.drawn_share else "photo"
        img, corners = render(drawn if source == "drawn" else alb, p, background=crop, centre=(cx, cy), rng=rng)
        x1, y1 = np.floor(corners.min(axis=0)).astype(int)
        x2, y2 = np.ceil(corners.max(axis=0)).astype(int) + 1
        x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, size), min(y2, size)
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        m = measure(img, (x1, y1, x2, y2), args.ring)
        rows.append(dict(i=i, source=source, **{f"p_{k}": v for k, v in asdict(p).items() if k != "extra"},
                         bx1=x1, by1=y1, bx2=x2, by2=y2, **m))
        if len(patches) < args.gallery_n:
            pad = 3
            patches.append(img[max(y1 - pad, 0):y2 + pad, max(x1 - pad, 0):x2 + pad])
    syn = pd.DataFrame(rows)
    syn["shape_ratio"] = syn.thickness / syn.length
    syn.round(3).to_csv(out / "synth_tags" / "synth_measurements.csv", index=False)
    log.info("rendered %d synthetic tags: looks %s, sources %s", len(syn), syn.p_look.value_counts().to_dict(),
             syn.source.value_counts().to_dict())

    real = pd.read_csv(args.real)
    cmp_rows = []
    for m, _ in COMPARE:
        for name, d in (("real", real), ("synthetic", syn)):
            q = np.nanpercentile(d[m], [0, 5, 50, 95, 100])
            cmp_rows.append(dict(measure=m, data=name, min=q[0], p5=q[1], median=q[2], p95=q[3], max=q[4]))
        lo, hi = np.nanpercentile(syn[m], [1, 99])
        inside = ((real[m] >= lo) & (real[m] <= hi)).mean()
        cmp_rows.append(dict(measure=m, data="real inside synthetic 1-99%", min=inside))
    cmp = pd.DataFrame(cmp_rows)
    cmp.round(3).to_csv(out / "synth_tags" / "compare.csv", index=False)
    log.info("real vs synthetic:\n%s", cmp.round(2).to_string(index=False))

    # figures
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.2))
    for ax, (m, lab) in zip(axes.ravel(), COMPARE):
        both = pd.concat([real[m], syn[m]]).dropna()
        bins = np.arange(-0.5, both.max() + 1.5) if m == "n_dips" else np.linspace(both.min(), both.max(), 25)
        ax.hist(syn[m].dropna(), bins=bins, density=True, color=SYN_COLOR, alpha=0.55, label="synthetic")
        ax.hist(real[m].dropna(), bins=bins, density=True, histtype="step", color=REAL_COLOR, lw=2, label="real")
        ax.set_title(lab, fontsize=9.5, loc="left")
        ax.tick_params(labelsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8)
    for ax in axes.ravel()[len(COMPARE):]:
        ax.axis("off")
    fig.suptitle("Synthetic tags (filled) should cover the real tags (outline)", fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(figs / "step3b_compare.png", dpi=130)
    plt.close(fig)

    # real tags for the gallery: cut from the video the same way the labelling tool did
    boxes = pd.read_csv(args.tag_boxes)
    cap = cv2.VideoCapture(str(SAMPLE_VIDEO))
    lens = Lens.for_frame(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    real_patches = []
    for r in boxes.iloc[np.linspace(0, len(boxes) - 1, min(args.gallery_n, len(boxes))).astype(int)].itertuples():
        cap.set(cv2.CAP_PROP_POS_FRAMES, r.frame)
        ok, fr = cap.read()
        v, _ = person_view(fr, (r.staff_x1, r.staff_y1, r.staff_x2, r.staff_y2), lens, r.margin)
        g = cv2.cvtColor(v.img, cv2.COLOR_BGR2GRAY)
        real_patches.append(g[max(r.y1 - 3, 0):r.y2 + 3, max(r.x1 - 3, 0):r.x2 + 3])
    cap.release()
    cv2.imwrite(str(figs / "step3b_gallery.png"), gallery(real_patches, patches, 64, 10))
    log.info("figures in %s (%.1f s)", figs, time.time() - t0)


if __name__ == "__main__":
    main()
