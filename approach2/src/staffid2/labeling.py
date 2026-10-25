"""Tracker-assisted labelling of one target (the staff member): the logic, no UI.

The user's decisions are stored as a list of actions (replayable, saved as JSON):
    {"type": "track",  "frame": f, "track_id": t}   target = track t from frame f until t ends
    {"type": "manual", "frame": f, "x": x, "y": y,  target at a clicked point (no track box there);
     "x1": .., "y1": .., "x2": .., "y2": ..}        box = size of the nearest track box, centred on the click
    {"type": "cut",    "frame": f, "track_id": t}   track t is NOT the target from frame f on
                                                    (fixes an ID swap)
    {"type": "absent", "frame": f}                  target not visible from f until the next label
    {"type": "ok",     "frame": s, "to": e}         interpolated frames s..e were checked and look right
                                                    (no effect on the labels; the tool stops asking)

Every action also carries "session" (the tool run that made it) and, when that run was limited
to a range, "scope": [first, last] frame. A scoped action only affects frames inside its scope.

build_labels() replays the actions session by session (later ones win) into one row per frame:
    source = track | manual | interp | absent | unlabelled
x, y is always the box centre. Runs of unlabelled frames of at most max_gap frames between two
frames with a position are filled by straight-line interpolation of the box, unless an "absent"
mark covers them. Longer runs stay "unlabelled" for the user to resolve.

A scoped session first clears the interpolation inside its scope, applies its actions clipped to
the scope, then re-fills absent and interpolated frames inside the scope only (frames outside may
serve as interpolation anchors but are never changed). So editing frames 800-850 cannot change
any label outside 800-850.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LABELLED = ("track", "manual")
PRESENT = ("track", "manual", "interp")


def _sessions(actions: list[dict]) -> list[tuple[tuple[int, int] | None, list[dict]]]:
    """Group consecutive actions with the same session and scope."""
    groups: list[tuple[tuple, tuple[int, int] | None, list[dict]]] = []
    for a in actions:
        scope = tuple(a["scope"]) if a.get("scope") else None
        key = (a.get("session"), scope)
        if groups and groups[-1][0] == key:
            groups[-1][2].append(a)
        else:
            groups.append((key, scope, [a]))
    return [(scope, acts) for _, scope, acts in groups]


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """(first, last) index of each run of True."""
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return []
    splits = np.flatnonzero(np.diff(idx) > 1)
    starts = np.r_[idx[0], idx[splits + 1]]
    ends = np.r_[idx[splits], idx[-1]]
    return list(zip(starts.tolist(), ends.tolist()))


def build_labels(actions: list[dict], tracks: pd.DataFrame, n_frames: int, max_gap: int) -> pd.DataFrame:
    """Replay actions into a per-frame table: frame, staff_present, x, y, x1, y1, x2, y2, source, track_id."""
    box = np.full((n_frames, 4), np.nan)
    src = np.array(["unlabelled"] * n_frames, dtype=object)
    tid = np.full(n_frames, -1)
    by_track = {t: g.sort_values("frame") for t, g in tracks.groupby("track_id")}

    for scope, acts in _sessions(actions):
        lo, hi = scope if scope else (0, n_frames - 1)
        inside = np.zeros(n_frames, bool)
        inside[lo:hi + 1] = True
        if scope:  # redo the interpolation inside the scope
            clear = inside & (src == "interp")
            box[clear], src[clear], tid[clear] = np.nan, "unlabelled", -1
        absent_starts = []
        for a in acts:
            f = int(a["frame"])
            if a["type"] == "track":
                g = by_track.get(int(a["track_id"]))
                if g is None:
                    continue
                g = g[(g.frame >= max(f, lo)) & (g.frame <= hi)]
                fr = g.frame.to_numpy()
                box[fr] = g[["x1", "y1", "x2", "y2"]].to_numpy()
                src[fr], tid[fr] = "track", int(a["track_id"])
            elif a["type"] == "manual" and lo <= f <= hi:
                box[f] = [a.get("x1", a["x"]), a.get("y1", a["y"]), a.get("x2", a["x"]), a.get("y2", a["y"])]
                src[f], tid[f] = "manual", -1
            elif a["type"] == "cut":
                sel = inside & (np.arange(n_frames) >= f) & (tid == int(a["track_id"])) & (src == "track")
                box[sel], src[sel], tid[sel] = np.nan, "unlabelled", -1
            elif a["type"] == "absent" and lo <= f <= hi:
                absent_starts.append(f)

        # absent: from the mark until the next labelled frame (within the scope)
        for f in absent_starts:
            k = f
            while k <= hi and src[k] not in LABELLED:
                src[k], box[k], tid[k] = "absent", np.nan, -1
                k += 1

        # interpolate short runs of unlabelled frames inside the scope
        for s, e in _runs(inside & (src == "unlabelled")):
            a, b = s - 1, e + 1
            if a < 0 or b >= n_frames or src[a] not in PRESENT or src[b] not in PRESENT:
                continue
            if b - a - 1 <= max_gap:
                t = (np.arange(s, e + 1) - a) / (b - a)
                box[s:e + 1] = box[a] + t[:, None] * (box[b] - box[a])
                src[s:e + 1] = "interp"

    df = pd.DataFrame(box, columns=["x1", "y1", "x2", "y2"])
    df.insert(0, "frame", np.arange(n_frames))
    df["x"], df["y"] = (df.x1 + df.x2) / 2, (df.y1 + df.y2) / 2
    df["source"], df["track_id"] = src, tid
    df["staff_present"] = np.where(np.isin(src, PRESENT), 1, np.where(src == "absent", 0, -1))  # -1 = unknown
    return df[["frame", "staff_present", "x", "y", "x1", "y1", "x2", "y2", "source", "track_id"]]


def next_issue(labels: pd.DataFrame, actions: list[dict], start: int, end: int,
               min_interp: int) -> tuple[str, int, int] | None:
    """The first thing in [start, end] that needs the user: ("unlabelled", frame, frame) or
    ("interp", first, last) for an interpolated run of at least min_interp frames that has not
    been marked ok. None if nothing is left."""
    ok = {(int(a["frame"]), int(a["to"])) for a in actions if a["type"] == "ok"}
    src = labels.source.to_numpy()
    issues = []
    un = np.flatnonzero(src[start:end + 1] == "unlabelled")
    if len(un):
        f = start + int(un[0])
        issues.append(("unlabelled", f, f))
    for s, e in _runs(src == "interp"):
        if e - s + 1 >= min_interp and s >= start and s <= end and (s, e) not in ok:
            issues.append(("interp", s, e))
            break
    return min(issues, key=lambda i: i[1]) if issues else None


def manual_box(labels: pd.DataFrame, frame: int, x: float, y: float) -> list[float]:
    """Box for a clicked point: the size of the nearest (in time) track box, centred on the click.
    Falls back to a point if there is no track box at all."""
    tr = labels[labels.source == "track"]
    if tr.empty:
        return [x, y, x, y]
    r = tr.iloc[int(np.argmin(np.abs(tr.frame.to_numpy() - frame)))]
    w, h = r.x2 - r.x1, r.y2 - r.y1
    return [round(x - w / 2, 1), round(y - h / 2, 1), round(x + w / 2, 1), round(y + h / 2, 1)]


def next_gap(labels: pd.DataFrame, start: int) -> int | None:
    """First unlabelled frame at or after start (None if everything is resolved)."""
    idx = labels.index[(labels.frame >= start) & (labels.source == "unlabelled")]
    return int(labels.frame[idx[0]]) if len(idx) else None


def _speed(xy: np.ndarray) -> float:
    """Mean movement in px/frame over consecutive points."""
    return float(np.hypot(*np.diff(xy, axis=0).T).mean()) if len(xy) > 1 else 0.0


def suggest(labels: pd.DataFrame, tracks: pd.DataFrame, gap_start: int, max_gap: int,
            base_dist: float, max_speed: float, k: int = 3, speed_frames: int = 5,
            gap_weight: float = 3.0) -> list[dict]:
    """Tracks that most likely continue the target after it was lost at gap_start.

    Candidates are tracks that START in (last labelled frame, gap_start + max_gap]: a track that
    already existed while the target was still labelled is someone else. A candidate's first
    box may be at most base_dist + max_speed * gap px from the last known position (a walking
    person crosses the fisheye centre at up to ~30 px/frame). Score, lower is better, sum of:
      distance / allowed distance      (near)
      gap_weight * gap / max_gap       (soon; weighted most: the earliest plausible
                                        re-appearance is the continuation)
      |its speed - target's speed| / max_speed   (moves like the target did, over speed_frames)
    Returns up to k candidates {track_id, frame, x, y, dist, gap, speed, score}, best first.
    """
    prev = labels[(labels.frame < gap_start) & labels.source.isin(LABELLED)]
    if prev.empty:
        return []
    last = prev.iloc[-1]
    target_speed = _speed(prev.tail(speed_frames)[["x", "y"]].to_numpy())
    lo, hi = int(last.frame) + 1, gap_start + max_gap
    starts = tracks.groupby("track_id").frame.min()
    cand_ids = starts[(starts >= lo) & (starts <= hi)].index
    out = []
    for tid in cand_ids:
        g = tracks[tracks.track_id == tid].sort_values("frame")
        r = g.iloc[0]
        gap = int(r.frame) - int(last.frame)
        dist = float(np.hypot(r.x - last.x, r.y - last.y))
        allowed = base_dist + max_speed * gap
        if dist > allowed:
            continue
        speed = _speed(g.head(speed_frames)[["x", "y"]].to_numpy())
        score = dist / allowed + gap_weight * gap / max(max_gap, 1) + abs(speed - target_speed) / max_speed
        out.append(dict(track_id=int(tid), frame=int(r.frame), x=float(r.x), y=float(r.y),
                        dist=dist, gap=gap, speed=speed, score=score))
    return sorted(out, key=lambda c: c["score"])[:k]


def track_at(tracks: pd.DataFrame, frame: int, x: float, y: float) -> int | None:
    """Track whose box at this frame contains (x, y); the one with the nearest centre if several."""
    f = tracks[(tracks.frame == frame) & (tracks.x1 <= x) & (x <= tracks.x2) & (tracks.y1 <= y) & (y <= tracks.y2)]
    if f.empty:
        return None
    return int(f.track_id.iloc[np.argmin(np.hypot(f.x - x, f.y - y))])


def jumps(labels: pd.DataFrame, max_px: float) -> pd.DataFrame:
    """Consecutive labelled frames whose centre moves more than max_px: possible ID swaps."""
    lab = labels[labels.staff_present == 1]
    d = np.hypot(lab.x.diff(), lab.y.diff())
    consecutive = lab.frame.diff() == 1
    return lab[(d > max_px) & consecutive].assign(jump_px=d[(d > max_px) & consecutive].round(1))
