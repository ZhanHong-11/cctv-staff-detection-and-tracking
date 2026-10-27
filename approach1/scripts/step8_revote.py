"""Step 8: compare voting rules on cached tracks + tag scores (seconds instead of minutes).

Uses the original video's results (data/step2_3, data/step4) and the cached
robustness variants (data/robustness/<name>_tracks.csv, <name>_tagscores.csv).
Each rule is a set of VoteParams overrides; nothing is re-detected.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from staffid.common import GT_DIR, ROOT, setup_logging  # noqa: E402
from staffid.vote import VoteParams, link_tracks, staff_frames, vote  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step8_robustness import VARIANTS, score  # noqa: E402

RULES = {
    "baseline": {},
    "no_near_links": {"near_dist": 0.0},
    "window_125": {"evidence_window": 125},
    "window_250": {"evidence_window": 250},
    "hit_rate_3pct": {"min_hit_rate": 0.03},
    "no_near+window_250": {"near_dist": 0.0, "evidence_window": 250},
}
N_FRAMES = 1341


def main() -> None:
    log = setup_logging("step8_revote")
    gt = pd.read_csv(GT_DIR / "ground_truth_draft.csv")
    rob = ROOT / "data" / "robustness"
    data = {"original": (pd.read_csv(ROOT / "data" / "step2_3" / "tracks.csv"),
                         pd.read_csv(ROOT / "data" / "step4" / "tag_scores.csv"), lambda x, y: (x, y))}
    for name, (_, inv) in VARIANTS.items():
        if (rob / f"{name}_tracks.csv").exists():
            data[name] = (pd.read_csv(rob / f"{name}_tracks.csv"), pd.read_csv(rob / f"{name}_tagscores.csv"), inv)

    rows = []
    for rule, over in RULES.items():
        p = replace(VoteParams(), **over)
        for name, (tracks, scores, inv) in data.items():
            chain = link_tracks(tracks, p)
            votes = vote(scores, chain, p)
            pred = staff_frames(tracks, chain, votes, N_FRAMES, p, scores)
            r = score(pred, gt, inv)
            rows.append({"rule": rule, "video": name, "recall": r["recall"], "within_25px": r["within_25px"],
                         "staff_chains": int(votes.staff.sum())})
    df = pd.DataFrame(rows)
    for col in ("recall", "within_25px"):
        log.info("%s:\n%s", col, df.pivot(index="rule", columns="video", values=col).reindex(RULES).round(3))
    df.to_csv(rob / "revote.csv", index=False)


if __name__ == "__main__":
    main()
