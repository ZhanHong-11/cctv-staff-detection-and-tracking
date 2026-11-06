"""Charts of the tag score over time: the staff member (answer key) vs a few other people.

Shows why one strong hit (confirmation threshold) works and counting hits does not: the staff
member's real tag gives rare, tall, 1-3 frame spikes; some still people give many medium bumps
that stay just below the confirmation line.

Inspection tool for the sample video: --tracks names the other people to plot (defaults: two
seated people with many hits, found with score_tracks.py's output).

Writes docs/figures/step3c_score_over_time.png and step3c_score_zoom.png.

Usage (from approach2/):
    ../.venv/bin/python scripts/score_charts.py
    ../.venv/bin/python scripts/score_charts.py --tracks 63 292 --zoom 360 570
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from staffid2.common import ANSWER_KEY, ROOT, TAG_SCORES, TAG_SIGHTINGS, setup_logging  # noqa: E402

GREY, ORANGE, GREEN, BLUE = "#8a8a85", "#eb8a1e", "#1a9e4a", "#2a78d6"


def on_timeline(frames: np.ndarray, scores: np.ndarray, lo: int, hi: int) -> pd.Series:
    """Scores on every frame lo..hi, NaN where there is no box (so the line breaks there)."""
    s = pd.Series(np.nan, index=range(lo, hi + 1))
    s[frames] = scores
    return s


def panel(ax, s: pd.Series, title: str, hit: float, conf: float, legend: bool = False) -> None:
    fr, sc = s.index.to_numpy(), s.to_numpy()
    ax.plot(fr, sc, color=GREY, lw=0.8, zorder=1)
    h, c = (sc >= hit) & (sc <= conf), sc > conf
    ax.scatter(fr[h], sc[h], s=14, color=ORANGE, zorder=3, label=f"hit (>= {hit})")
    ax.scatter(fr[c], sc[c], s=26, color=GREEN, zorder=4, label=f"confirmation (> {conf})")
    ax.axhline(hit, color=ORANGE, ls="--", lw=0.9)
    ax.axhline(conf, color=GREEN, ls="--", lw=0.9)
    ax.axhline(0, color="#cfcfca", lw=0.8)
    ax.set_title(title, loc="left", fontsize=10)
    ax.set_ylim(-0.13, 0.12)
    ax.set_ylabel("tag score")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    if legend:
        ax.legend(loc="upper right", fontsize=8, frameon=False, ncol=2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", default=str(TAG_SCORES))
    ap.add_argument("--answer-key", default=str(ANSWER_KEY))
    ap.add_argument("--sightings", default=str(TAG_SIGHTINGS))
    ap.add_argument("--tracks", type=int, nargs="+", default=[63, 292], help="other people to plot")
    ap.add_argument("--zoom", type=int, nargs=2, default=[360, 570], help="staff frames for the zoomed chart")
    ap.add_argument("--hit-thr", type=float, default=0.037)
    ap.add_argument("--conf-thr", type=float, default=0.065)
    ap.add_argument("--fig-dir", default=str(ROOT / "docs" / "figures"))
    args = ap.parse_args()

    log = setup_logging("score_charts")
    log.info("params: %s", vars(args))
    figs = Path(args.fig_dir)
    figs.mkdir(parents=True, exist_ok=True)
    s = pd.read_csv(args.scores)
    key = pd.read_csv(args.answer_key).set_index("frame")
    n = int(max(s.frame.max(), key.index.max())) + 1
    staff = s[s.track_id.to_numpy() == key.track_id.reindex(s.frame).to_numpy()].sort_values("frame")
    hit, conf = args.hit_thr, args.conf_thr

    fig, axes = plt.subplots(1 + len(args.tracks), 1, figsize=(13, 3 + 3 * len(args.tracks)),
                             gridspec_kw={"height_ratios": [1.3] + [1] * len(args.tracks)})
    panel(axes[0], on_timeline(staff.frame.to_numpy(), staff.score.to_numpy(), 0, n - 1),
          "STAFF member (answer-key box): gaps = no tracker box", hit, conf, legend=True)
    if Path(args.sightings).exists():
        for _, r in pd.read_csv(args.sightings).iterrows():
            axes[0].axvspan(r.start, r.end, color=BLUE, alpha=0.12, lw=0)
        axes[0].text(0, 0.105, "blue bands = the user's tag sightings", fontsize=8, color=BLUE)
    for ax, tid in zip(axes[1:], args.tracks):
        g = s[s.track_id == tid].sort_values("frame")
        moved = float(np.hypot(g.x.max() - g.x.min(), g.y.max() - g.y.min()))
        nh = int((g.score >= hit).sum())
        log.info("track %d: %d frames, %d hits, max %.3f, moved %.0f px", tid, len(g), nh, g.score.max(), moved)
        panel(ax, on_timeline(g.frame.to_numpy(), g.score.to_numpy(), 0, n - 1),
              f"NOT staff: track {tid} ({len(g)} frames, box centre moved {moved:.0f} px): {nh} hits, "
              f"highest {g.score.max():.3f}", hit, conf)
    axes[-1].set_xlabel("frame")
    fig.tight_layout()
    fig.savefig(figs / "step3c_score_over_time.png", dpi=110)
    plt.close(fig)

    lo, hi = args.zoom
    z = staff[(staff.frame >= lo) & (staff.frame <= hi)]
    g = s[s.track_id == args.tracks[-1]].sort_values("frame")
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.6), sharey=True)
    panel(axes[0], on_timeline(z.frame.to_numpy(), z.score.to_numpy(), lo, hi), f"STAFF, frames {lo}-{hi}", hit, conf)
    panel(axes[1], on_timeline(g.frame.to_numpy(), g.score.to_numpy(), int(g.frame.min()), int(g.frame.max())),
          f"NOT staff: track {args.tracks[-1]}, whole track", hit, conf)
    axes[1].set_ylabel("")
    for ax in axes:
        ax.set_xlabel("frame")
    fig.tight_layout()
    fig.savefig(figs / "step3c_score_zoom.png", dpi=110)
    plt.close(fig)
    log.info("wrote %s, %s", figs / "step3c_score_over_time.png", figs / "step3c_score_zoom.png")


if __name__ == "__main__":
    main()
