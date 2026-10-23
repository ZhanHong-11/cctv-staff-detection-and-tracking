"""Step 5: decide which people are staff by voting over whole tracks.

The tag is visible in only a fraction of frames, so single frames are not
enough. Instead:
1. link_tracks(): join track fragments into chains. Tracks break where the
   detector loses people (mainly at the fisheye centre); a fragment that ends
   and another that starts shortly after, close enough to be reached at walking
   speed, are treated as the same person.
2. vote(): a chain is staff if it collects at least `min_hits` tag hits.
3. staff_frames(): per frame, the staff box (from tracks, or interpolated in the
   gaps inside a staff chain).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class VoteParams:
    hit_thresh: float = 0.9     # tag score that counts as a hit
    min_hits: int = 2           # hits a chain needs to be called staff
    max_overlap: int = 3        # fragments may overlap slightly (ID switch)
    # near link: the person stayed roughly in place (e.g. a seated person whose track broke)
    near_dist: float = 60.0     # px
    near_gap: int = 40          # frames
    # jump link: the person walked through a blind spot (mainly the fisheye centre); the jump
    # must continue the direction the first fragment was moving and match how the next one starts
    jump_gap: int = 25          # frames
    jump_speed: float = 35.0    # max px per frame of gap (observed hand-offs: 21-33)
    min_cos: float = 0.8        # cos(angle) between end/start motion and the jump (~37 deg)
    min_motion: float = 1.0     # px/frame; slower fragments can't make jump links
    vel_frames: int = 8         # frames used to estimate end/start velocity
    fill_gaps: bool = True      # interpolate staff position inside a staff chain


def track_table(tracks: pd.DataFrame, vel_frames: int = 8) -> pd.DataFrame:
    """Per track: first/last frame and position, plus velocity at the start and at the end."""
    t = tracks.assign(cx=(tracks.x1 + tracks.x2) / 2, cy=(tracks.y1 + tracks.y2) / 2).sort_values("frame")
    rows = {}
    for tid, g in t.groupby("track_id"):
        f, x, y = g.frame.to_numpy(), g.cx.to_numpy(), g.cy.to_numpy()
        k = min(vel_frames, len(f) - 1)
        def vel(i0, i1):
            dt = max(f[i1] - f[i0], 1)
            return (x[i1] - x[i0]) / dt, (y[i1] - y[i0]) / dt
        svx, svy = vel(0, k) if k > 0 else (0.0, 0.0)
        evx, evy = vel(-1 - k, -1) if k > 0 else (0.0, 0.0)
        rows[tid] = dict(first=f[0], last=f[-1], fx=x[0], fy=y[0], lx=x[-1], ly=y[-1],
                         svx=svx, svy=svy, evx=evx, evy=evy)
    return pd.DataFrame.from_dict(rows, orient="index")


def _cos(ax: float, ay: float, bx: float, by: float) -> float:
    return (ax * bx + ay * by) / (np.hypot(ax, ay) * np.hypot(bx, by) + 1e-9)


def link_tracks(tracks: pd.DataFrame, p: VoteParams) -> dict[int, int]:
    """Return track_id -> chain_id. Greedy matching of track ends to track starts by cost."""
    tt = track_table(tracks, p.vel_frames)
    ids = tt.index.to_numpy()
    max_gap = max(p.near_gap, p.jump_gap)
    pairs = []
    for a in ids:
        A = tt.loc[a]
        cand = tt[(tt["first"] > A["last"] - p.max_overlap) & (tt["first"] <= A["last"] + max_gap)
                  & (tt["last"] > A["last"])]
        for b, B in cand.iterrows():
            if b == a:
                continue
            gap = max(B["first"] - A["last"], 1)
            dx, dy = B.fx - A.lx, B.fy - A.ly
            dist = float(np.hypot(dx, dy))
            if dist <= p.near_dist and gap <= p.near_gap:
                pairs.append((dist / p.near_dist + gap / p.near_gap, a, b))
                continue
            if gap > p.jump_gap or dist > p.jump_speed * gap:
                continue
            if np.hypot(A.evx, A.evy) < p.min_motion or np.hypot(B.svx, B.svy) < p.min_motion:
                continue
            ca, cb = _cos(A.evx, A.evy, dx, dy), _cos(B.svx, B.svy, dx, dy)
            if ca >= p.min_cos and cb >= p.min_cos:
                # jump links rank after near links; better-aligned and shorter gaps first
                pairs.append((2 + (2 - ca - cb) + gap / p.jump_gap, a, b))
    nxt, prv = {}, {}
    for _, a, b in sorted(pairs):
        if a not in nxt and b not in prv:
            nxt[a], prv[b] = b, a
    chain = {}
    for t in ids:
        if t in prv:
            continue
        cid, cur = int(t), t
        while cur is not None:
            chain[int(cur)] = cid
            cur = nxt.get(cur)
    return chain


def vote(scores: pd.DataFrame, chain: dict[int, int], p: VoteParams) -> pd.DataFrame:
    """Per chain: number of scored boxes, hits, best score, and the staff decision."""
    s = scores.assign(chain=scores.track_id.map(chain), hit=scores.score >= p.hit_thresh)
    v = s.groupby("chain").agg(scored=("score", "size"), hits=("hit", "sum"), best=("score", "max"))
    v["staff"] = v.hits >= p.min_hits
    return v


def staff_frames(tracks: pd.DataFrame, chain: dict[int, int], votes: pd.DataFrame, n_frames: int,
                 p: VoteParams) -> pd.DataFrame:
    """One row per frame: staff_present, x, y (box centre), box, source (track/interp/none)."""
    t = tracks.assign(chain=tracks.track_id.map(chain))
    staff_chains = votes.index[votes.staff]
    t = t[t.chain.isin(staff_chains)]
    out = pd.DataFrame({"frame": np.arange(n_frames), "staff_present": 0, "x": np.nan, "y": np.nan,
                        "x1": np.nan, "y1": np.nan, "x2": np.nan, "y2": np.nan, "source": "none",
                        "chain": -1})
    if t.empty:
        return out
    # if several staff chains share a frame, keep the one with the most hits
    t = t.assign(hits=t.chain.map(votes.hits)).sort_values(["frame", "hits"], ascending=[True, False])
    t = t.drop_duplicates("frame")
    f = t.frame.to_numpy()
    out.loc[f, ["x1", "y1", "x2", "y2"]] = t[["x1", "y1", "x2", "y2"]].to_numpy()
    out.loc[f, "chain"] = t.chain.to_numpy()
    out.loc[f, "source"] = "track"
    if p.fill_gaps:
        for c, g in t.groupby("chain"):
            lo, hi = g.frame.min(), g.frame.max()
            missing = np.setdiff1d(np.arange(lo, hi + 1), g.frame.to_numpy())
            missing = missing[out.loc[missing, "source"].to_numpy() == "none"]
            if len(missing) == 0:
                continue
            for col in ("x1", "y1", "x2", "y2"):
                out.loc[missing, col] = np.interp(missing, g.frame, g[col])
            out.loc[missing, "source"] = "interp"
            out.loc[missing, "chain"] = c
    present = out.source != "none"
    out.loc[present, "staff_present"] = 1
    out["x"], out["y"] = (out.x1 + out.x2) / 2, (out.y1 + out.y2) / 2
    return out
