"""Step 3c: evaluate the NCC tag detector on person crops, against the user's tag labels.

Crops (rectified, gray): the staff member in every frame (answer-key box) and other people
every --neg-every frames (tracker boxes, not overlapping the staff box). Each crop gets the best
NCC score over the template bank, inside the chest area.

Honest threshold: negatives are split by track ID into half A and half B. The threshold is the
score that only --fp-target of half-A crops exceed; all false-hit rates are reported on half B.
Template sources are only the brief's photo and a drawn tag (step 3b); no video tags are used.

Metrics (per configuration)
- hit rate on staff frames inside the user's sightings (tag visible), per frame and per sighting
- on-tag rate: on the 63 frames with a tag box, the best match lies on the box (+-tol px)
- false-hit rate on negatives (half B), and the worst negative track's hit rate (matters for
  voting in step 4: a track is staff if it hits often; a non-staff track must not)
- staff hit rate outside sightings (tag not labelled there: back turned, too blurry, skipped)

Writes outputs/ncc_eval/{crops_index,scores_<config>,metrics,per_sighting}.csv and
docs/figures/step3c_*.png.

Usage (from approach2/):
    ../.venv/bin/python scripts/ncc_eval.py                 # all configurations
    ../.venv/bin/python scripts/ncc_eval.py --configs base  # one
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from staffid2.common import (ANSWER_KEY, BRIEF_PDF, ROOT, SAMPLE_VIDEO, TAG_BOXES,  # noqa: E402
                                    TAG_SIGHTINGS, TRACKS_CSV, setup_logging)
from staffid2.tag_score import gray_crops, score_crops  # noqa: E402
from staffid2.ncc import BankConfig, MatchConfig, build_bank  # noqa: E402
from staffid2.tag_source import draw_tag, find_tag, measure_layout, pdf_jpegs, to_albedo  # noqa: E402

BASE_BANK, BASE_MATCH = BankConfig(), MatchConfig()
CONFIGS = {
    "base": (BASE_BANK, BASE_MATCH),                                   # drawn tag, no flip, chest area, verify
    "raw": (BASE_BANK, replace(BASE_MATCH, mode="raw")),               # plain best NCC (first try)
    "comb": (BASE_BANK, replace(BASE_MATCH, mode="comb")),             # comb maximised everywhere (second try)
    "verify_half": (BASE_BANK, replace(BASE_MATCH, twin_weight=0.5)),
    "photo": (replace(BASE_BANK, sources=("photo",)), BASE_MATCH),     # the photo's tag instead
    "both": (replace(BASE_BANK, sources=("drawn", "photo")), BASE_MATCH),
    "flip": (replace(BASE_BANK, flips=(False, True)), BASE_MATCH),     # also upside-down tags
    "no_mid": (replace(BASE_BANK, looks=("light", "dark")), BASE_MATCH),
    "full_crop": (BASE_BANK, replace(BASE_MATCH, region=(0.0, 0.0, 1.0, 1.0))),
}
COLORS = {"staff, tag visible": "#2a78d6", "staff, not labelled": "#9a9a94", "other people (B)": "#eb6834"}


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of boxes a (N, 4) with one box b (4,)."""
    ix = np.clip(np.minimum(a[:, 2], b[2]) - np.maximum(a[:, 0], b[0]), 0, None)
    iy = np.clip(np.minimum(a[:, 3], b[3]) - np.maximum(a[:, 1], b[1]), 0, None)
    inter = ix * iy
    return inter / ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def crop_table(neg_every: int, min_side: float, max_iou: float) -> pd.DataFrame:
    key = pd.read_csv(ANSWER_KEY)
    key = key[(key.staff_present == 1) & key.x1.notna()]
    sights = pd.read_csv(TAG_SIGHTINGS)
    boxes = pd.read_csv(TAG_BOXES).set_index("frame")
    staff = key[["frame", "x1", "y1", "x2", "y2", "source", "track_id"]].copy()
    staff["role"] = "staff"
    staff["sighting"] = -1
    for s in sights.itertuples():
        staff.loc[staff.frame.between(s.start, s.end), "sighting"] = s.sighting
    for c in ("bx1", "by1", "bx2", "by2"):
        staff[c] = np.nan
    has = staff.frame.isin(boxes.index)
    staff.loc[has, ["bx1", "by1", "bx2", "by2"]] = boxes.loc[staff.frame[has], ["x1", "y1", "x2", "y2"]].to_numpy()

    tracks = pd.read_csv(TRACKS_CSV)
    tracks = tracks[tracks.frame % neg_every == 0]
    keyb = key.set_index("frame")
    keep = []
    for f, g in tracks.groupby("frame"):
        b = g[["x1", "y1", "x2", "y2"]].to_numpy(float)
        ok = (np.minimum(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]) >= min_side)
        if f in keyb.index:
            k = keyb.loc[f]
            ok &= (g.track_id.to_numpy() != k.track_id) & (iou(b, np.array([k.x1, k.y1, k.x2, k.y2])) <= max_iou)
        keep.append(g[ok])
    neg = pd.concat(keep)[["frame", "track_id", "x1", "y1", "x2", "y2"]].copy()
    neg["role"] = "neg"
    neg["half"] = np.where(neg.track_id % 2 == 0, "A", "B")
    return pd.concat([staff, neg], ignore_index=True)


def metrics(t: pd.DataFrame, fp_target: float, tol: float) -> tuple[dict, pd.DataFrame]:
    negA, negB = t[(t.role == "neg") & (t.half == "A")], t[(t.role == "neg") & (t.half == "B")]
    thr = float(np.quantile(negA.score, 1 - fp_target))
    st = t[t.role == "staff"]
    vis, out = st[st.sighting >= 0], st[st.sighting < 0]
    hit = lambda d: float((d.score >= thr).mean()) if len(d) else float("nan")  # noqa: E731
    boxed = vis[vis.bx1.notna()]
    on = ((boxed.score >= thr) & (boxed.mx >= boxed.bx1 - tol) & (boxed.mx <= boxed.bx2 + tol)
          & (boxed.my >= boxed.by1 - tol) & (boxed.my <= boxed.by2 + tol))
    per_s = vis.groupby("sighting").apply(lambda g: pd.Series(dict(
        frames=len(g), hits=int((g.score >= thr).sum()), hit_rate=float((g.score >= thr).mean()),
        best=float(g.score.max())))).reset_index()
    per_track = negB.groupby("track_id").score.apply(lambda s: (s >= thr).mean())
    per_track = per_track[negB.groupby("track_id").size() >= 10]
    m = dict(threshold=thr, hit_visible=hit(vis), sightings_hit=int((per_s.hits > 0).sum()), sightings=len(per_s),
             on_tag=float(on.mean()), on_tag_n=int(on.sum()), boxed=len(boxed), fp_B=hit(negB),
             fp_worst_track_B=float(per_track.max()) if len(per_track) else float("nan"),
             fp_track_p90_B=float(per_track.quantile(0.9)) if len(per_track) else float("nan"),
             hit_staff_all=hit(st), hit_not_labelled=hit(out), n_vis=len(vis), n_negB=len(negB))
    return m, per_s


def roc(t: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Hit rate on visible-tag staff frames vs false-hit rate on negatives B, over all thresholds."""
    vis = t[(t.role == "staff") & (t.sighting >= 0)].score.to_numpy()
    negB = t[(t.role == "neg") & (t.half == "B")].score.to_numpy()
    th = np.unique(np.concatenate([vis, negB]))[::-1]
    return np.array([(negB >= x).mean() for x in th]), np.array([(vis >= x).mean() for x in th])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    ap.add_argument("--main", default="base", help="configuration used for the detailed figures")
    ap.add_argument("--neg-every", type=int, default=5, help="negatives: every Nth frame")
    ap.add_argument("--min-side", type=float, default=20.0)
    ap.add_argument("--max-iou", type=float, default=0.3, help="skip negatives overlapping the staff box more")
    ap.add_argument("--margin", type=float, default=0.15, help="crop margin (as staff_crops.py / label_tag.py)")
    ap.add_argument("--fp-target", type=float, default=0.02, help="threshold: this fraction of negatives A pass")
    ap.add_argument("--tol", type=float, default=2.0, help="on-tag check tolerance (px)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "ncc_eval"))
    ap.add_argument("--fig-dir", default=str(ROOT / "docs" / "figures"))
    args = ap.parse_args()

    log = setup_logging("ncc_eval")
    log.info("params: %s", vars(args))
    t0 = time.time()
    out, figs = Path(args.out), Path(args.fig_dir)
    out.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    photo = pdf_jpegs(BRIEF_PDF)[0]
    alb = to_albedo(photo, find_tag(photo))
    drawn = draw_tag(measure_layout(alb), 120)
    albedos = {"photo": alb, "drawn": drawn}
    layouts = {"photo": measure_layout(alb), "drawn": measure_layout(drawn)}

    table = crop_table(args.neg_every, args.min_side, args.max_iou)
    table.to_csv(out / "crops_index.csv", index=False)
    log.info("crops: staff %d (tag visible %d, boxed %d), negatives %d from %d tracks (A %d / B %d crops)",
             (table.role == "staff").sum(), ((table.role == "staff") & (table.sighting >= 0)).sum(),
             table.bx1.notna().sum(), (table.role == "neg").sum(), table[table.role == "neg"].track_id.nunique(),
             ((table.role == "neg") & (table.half == "A")).sum(), ((table.role == "neg") & (table.half == "B")).sum())
    crops = gray_crops(str(SAMPLE_VIDEO), table, args.margin)
    log.info("cut %d crops (%.1f s)", len(crops), time.time() - t0)

    rows, results = [], {}
    for name in args.configs:
        bcfg, mcfg = CONFIGS[name]
        t1 = time.time()
        sc = score_crops(crops, albedos, layouts, bcfg, mcfg, args.workers)
        dt = time.time() - t1
        t = pd.concat([table.reset_index(drop=True), sc], axis=1)
        t.round(3).to_csv(out / f"scores_{name}.csv", index=False)
        m, per_s = metrics(t, args.fp_target, args.tol)
        n_t = len(build_bank(albedos, layouts, bcfg))
        rows.append(dict(config=name, templates=n_t, ms_per_crop=1000 * dt * (args.workers or 1) / len(crops),
                         seconds=dt, **m))
        results[name] = (t, per_s)
        log.info("%s: %d templates, %.1f s; %s", name, n_t, dt, {k: round(v, 3) if isinstance(v, float) else v
                                                                  for k, v in m.items()})
    met = pd.DataFrame(rows)
    met.round(4).to_csv(out / "metrics.csv", index=False)
    log.info("metrics:\n%s", met.round(3).to_string(index=False))

    # ---------------------------------------------------------------- figures (main config)
    t, per_s = results[args.main if args.main in results else args.configs[0]]
    per_s.to_csv(out / "per_sighting.csv", index=False)
    thr = met.set_index("config").loc[args.main if args.main in results else args.configs[0], "threshold"]
    groups = {"staff, tag visible": t[(t.role == "staff") & (t.sighting >= 0)].score,
              "staff, not labelled": t[(t.role == "staff") & (t.sighting < 0)].score,
              "other people (B)": t[(t.role == "neg") & (t.half == "B")].score}
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
    bins = np.linspace(min(g.quantile(0.005) for g in groups.values()), max(g.max() for g in groups.values()), 40)
    for lab, g in groups.items():
        axes[0].hist(g, bins=bins, density=True, histtype="step", lw=2, color=COLORS[lab], label=f"{lab} ({len(g)})")
    axes[0].axvline(thr, color="#33332f", ls="--", lw=1)
    axes[0].text(thr, axes[0].get_ylim()[1] * 0.95, f" threshold {thr:.2f}", fontsize=8, va="top")
    axes[0].set(xlabel="best score in the crop", ylabel="density")
    axes[0].set_title(f"Score per crop ({args.main})", fontsize=10, loc="left")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    for name, (tt, _) in results.items():
        fp, tp = roc(tt)
        axes[1].plot(fp, tp, lw=2 if name == args.main else 1.2, label=name)
    axes[1].set(xscale="log", xlim=(1e-3, 1), ylim=(0, 1), xlabel="false-hit rate on other people (B, log)",
                ylabel="hit rate on staff frames with tag visible")
    axes[1].set_title("Trade-off per configuration", fontsize=10, loc="left")
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.grid(True, color="#e4e4e0", lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(figs / "step3c_scores.png", dpi=130)
    plt.close(fig)

    # examples: hits on visible tags, misses, and the strongest false hits on other people
    def tile(i: int, title: str) -> np.ndarray:
        g = cv2.cvtColor(crops[i], cv2.COLOR_GRAY2BGR)
        r = t.iloc[i]
        if r.template >= 0:
            cv2.circle(g, (int(round(r.mx)), int(round(r.my))), 6, (0, 0, 255), 1)
        if pd.notna(r.get("bx1", np.nan)):
            cv2.rectangle(g, (int(r.bx1), int(r.by1)), (int(r.bx2) - 1, int(r.by2) - 1), (0, 200, 0), 1)
        g = cv2.resize(g, (180, 180), interpolation=cv2.INTER_AREA)
        cv2.rectangle(g, (0, 0), (180, 16), (0, 0, 0), -1)
        cv2.putText(g, title, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
        return g

    vis_i = t.index[(t.role == "staff") & (t.sighting >= 0) & t.bx1.notna()]
    hits = [i for i in vis_i if t.score[i] >= thr][:8]
    miss = [i for i in vis_i if t.score[i] < thr][:8]
    fps = t[(t.role == "neg") & (t.half == "B")].score.nlargest(8).index.tolist()
    lines = []
    for lab, idx in (("hit", hits), ("miss", miss), ("false hit", fps)):
        tiles = [tile(i, f"{lab} f{int(t.frame[i])} {t.score[i]:.2f}") for i in idx]
        tiles += [np.zeros((180, 180, 3), np.uint8)] * (8 - len(tiles))
        lines.append(np.hstack(tiles))
    cv2.imwrite(str(figs / "step3c_examples.png"), np.vstack(lines))
    log.info("done (%.1f s)", time.time() - t0)


if __name__ == "__main__":
    main()
