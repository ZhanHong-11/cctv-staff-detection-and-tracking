"""Step 4: live following of the staff member (accepted by the user 2026-10-26).

Rules: docs/step4_rules.md (short: docs/tag_recognition_&_staff_tracking.md). Frame t is decided from frames <= t only: the
Follower gets one frame's tracked boxes (track_id, box, tag score) at a time and keeps its own
memory (targets, when each track was first seen, recent positions).

Target states:
    CONFIRMED  the tag was seen on its current track (a confirmation, or a hit while UNCERTAIN)
    UNCERTAIN  following a continuation track chosen by motion; no tag seen on it yet
    LOST       its track ended; searching for a continuation (shown as a red cross)
    (dropped)  lost longer than search_time: removed

Step 4.5 case 1 (ID swap / crowded loss): when a target is lost, every moving person near the
loss point becomes a SUSPECT (old tracks included, unlike the continuation search); the first
hit on a suspect moves the target there. Centre / standing still: not yet.
After the run, fill_gaps() fills short LOST stretches between two pieces of the same target
(the saved CSV only; the live output is never filled).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ACTIVE = ("CONFIRMED", "UNCERTAIN")


@dataclass
class FollowConfig:
    conf_thr: float = 0.065        # confirmation: counts anywhere
    hit_thr: float = 0.037         # hit: counts only on an UNCERTAIN target's own track
    search_time: int = 62          # frames a LOST target is searched for (2.5 s)
    base_dist: float = 100.0       # continuation: allowed distance = base_dist + max_speed * gap
    max_speed: float = 30.0        # px/frame
    gap_weight: float = 3.0        # weight of "soon" in the continuation score
    speed_frames: int = 5          # frames used for a speed
    trust_frames: int = 10         # a guessed track followed this long becomes the new loss point
    fill_max: int = 62             # end fill: longest LOST stretch filled (frames)
    still_speed: float = 2.0       # end fill: slower than this before the loss ...
    still_dist: float = 40.0       # ... and found again this close -> "held" instead of "interp"
    focus_time: int = 25           # viewer only: a hit stays orange this long
    suspects: bool = True          # 4.5: suspects after a loss (False = 4.2 behaviour)
    suspect_radius: float = 250.0  # people this close to the loss point become suspects
    suspect_time: int = 125        # suspects are kept this long after the loss (5 s)
    suspect_move_px: float = 40.0  # a suspect's hit counts only if it moved this much ...
    move_frames: int = 25          # ... over the last move_frames (seated people's false hits)
    suspects_until_confirmed: bool = True   # keep suspects after a suspect hit (latest hit wins) until
                                            # a confirmation or suspect_time; False = first hit wins


@dataclass
class Target:
    tid: int                       # target number (T1, T2, ...)
    state: str
    track_id: int
    box: np.ndarray                # last known box x1, y1, x2, y2
    last_frame: int                # last frame with a box
    last_conf: int                 # frame of the last confirmation (> conf_thr); picks the main target
    reason: str
    hist: deque = field(default_factory=lambda: deque(maxlen=6))   # (frame, x, y)
    state_before_loss: str = "CONFIRMED"
    lost_at: int = -1
    loss: tuple | None = None      # (last frame, x, y, speed, track) when the track last ended
    on_track: int = 0              # frames with a box on the current track
    counted_track: int = -1        # the track on_track counts
    suspects: set = field(default_factory=set)   # 4.5: tracks where a hit moves this target
    suspect_since: int | None = None             # frame the suspect window opened

    @property
    def xy(self) -> tuple[float, float]:
        return float((self.box[0] + self.box[2]) / 2), float((self.box[1] + self.box[3]) / 2)

    def mark_loss(self, f: int, cfg: "FollowConfig") -> None:
        """Remember where the track ended. An UNCERTAIN target whose guessed track was followed only
        briefly (< trust_frames) and soon after the real loss keeps the earlier loss point: the
        guess's position is not trusted yet."""
        if (self.state_before_loss == "UNCERTAIN" and self.loss is not None and f - self.loss[0] <= cfg.search_time
                and self.on_track < cfg.trust_frames):
            return
        self.loss = (self.last_frame, *self.xy, _speed(self.hist), self.track_id)


def _speed(pts) -> float:
    """Mean movement in px/frame over consecutive (frame, x, y) points."""
    p = np.asarray(list(pts), float)
    if len(p) < 2:
        return 0.0
    return float(np.hypot(np.diff(p[:, 1]), np.diff(p[:, 2])).sum() / max(p[-1, 0] - p[0, 0], 1))


class Follower:
    """Feed frames in order with step(frame, rows); rows has track_id, x1, y1, x2, y2, score."""

    def __init__(self, cfg: FollowConfig):
        self.cfg = cfg
        self.targets: list[Target] = []
        self.next_tid = 1
        self.first: dict[int, tuple[int, float, float]] = {}    # track -> (first frame, x, y)
        self.early: dict[int, list] = {}                           # track -> first speed_frames points
        self.recent: dict[int, deque] = {}                         # track -> last move_frames points
        self.events: list[dict] = []

    # ---------------------------------------------------------------- helpers
    def _event(self, f: int, t: Target, what: str, reason: str) -> None:
        t.reason = reason
        self.events.append(dict(frame=f, target=t.tid, event=what, track_id=t.track_id, reason=reason))

    def _owned(self, exclude: Target | None = None) -> set[int]:
        return {t.track_id for t in self.targets if t.state in ACTIVE and t is not exclude}

    def _continuation(self, t: Target, f: int, present: dict[int, pd.Series], taken: set[int]):
        """Best (cost, track, text) continuing target t after its last loss, at frame f, or None.
        Only tracks first seen after the loss (a track that existed meanwhile is someone else),
        or the lost track itself coming back. A candidate's speed is measured over its boxes seen
        so far (up to speed_frames); with a single box it is unknown and not counted yet."""
        c = self.cfg
        lf, lx, ly, l_speed, l_track = t.loss
        if l_track in present and l_track not in taken:
            return 0.0, l_track, f"track {l_track} came back after {f - lf} frames"
        best = None
        for tr in present:
            if tr in taken:
                continue
            f0, x0, y0 = self.first[tr]
            if f0 <= lf:
                continue
            gap = f0 - lf
            dist = float(np.hypot(x0 - lx, y0 - ly))
            allowed = c.base_dist + c.max_speed * gap
            if dist > allowed or gap > c.search_time:
                continue
            early = self.early[tr]
            speed_term = abs(_speed(early) - l_speed) / c.max_speed if len(early) > 1 else 0.0
            cost = dist / allowed + c.gap_weight * gap / c.search_time + speed_term
            if best is None or cost < best[0]:
                best = (cost, tr, f"continues track {l_track}: new track {tr}, {gap} frames later, "
                                  f"{dist:.0f} px away (cost {cost:.2f})")
        return best

    def _revisable(self, t: Target, f: int) -> bool:
        """An UNCERTAIN target may still switch to a better continuation (no tag seen on it yet,
        loss recent)."""
        return t.state == "UNCERTAIN" and t.loss is not None and f - t.loss[0] <= self.cfg.search_time

    # ---------------------------------------------------------------- one frame
    def step(self, f: int, rows: pd.DataFrame) -> list[dict]:
        c = self.cfg
        present = {int(r.track_id): r for _, r in rows.iterrows()}
        for tr, r in present.items():
            if tr not in self.first:
                self.first[tr], self.early[tr] = (f, float(r.x), float(r.y)), []
            if len(self.early[tr]) < c.speed_frames:
                self.early[tr].append((f, float(r.x), float(r.y)))
            self.recent.setdefault(tr, deque(maxlen=c.move_frames)).append((float(r.x), float(r.y)))

        # 1. update targets on their tracks
        for t in self.targets:
            if t.state not in ACTIVE:
                continue
            if t.track_id in present:
                self._move(t, f, present[t.track_id])
            else:
                t.state_before_loss, t.state, t.lost_at = t.state, "LOST", f
                t.mark_loss(f, c)
                self._event(f, t, "lost", f"track {t.track_id} ended (last seen {t.last_frame})")
                if c.suspects and t.suspect_since is None:
                    t.suspect_since = f

        # 2a. an UNCERTAIN target switches if a better continuation of its loss has appeared
        for t in self.targets:
            if self._revisable(t, f):
                cont = self._continuation(t, f, present, self._owned(exclude=t))
                if cont is not None and cont[1] != t.track_id:
                    cur = self._continuation(t, f, {t.track_id: present[t.track_id]}, set())
                    if cur is None or cont[0] < cur[0]:
                        t.track_id = cont[1]
                        self._move(t, f, present[cont[1]])
                        self._event(f, t, "switched", cont[2])

        # 2b. search for lost targets (oldest loss first)
        for t in sorted([t for t in self.targets if t.state == "LOST"], key=lambda t: t.lost_at):
            cont = self._continuation(t, f, present, self._owned())
            if cont is not None:
                _, tr, text = cont
                revived = tr == t.loss[4]
                t.track_id, t.state = tr, (t.state_before_loss if revived else "UNCERTAIN")
                self._move(t, f, present[tr])
                self._event(f, t, "revived" if revived else "continuation", text)
            elif f - t.last_frame > c.search_time:
                t.state = "DROPPED"
                self._event(f, t, "dropped", f"nothing found within {c.search_time} frames")
        self.targets = [t for t in self.targets if t.state != "DROPPED"]

        # 2c. suspects: moving or not, old or new, everyone near the loss point while the window is open
        for t in self.targets:
            if t.suspect_since is None:
                continue
            if f - t.suspect_since > c.suspect_time:
                self._clear_suspects(t)
                continue
            _, lx, ly, _, _ = t.loss
            others = self._owned(exclude=t)
            new = {tr for tr, r in present.items() if tr not in others and tr != t.track_id
                   and tr not in t.suspects and np.hypot(r.x - lx, r.y - ly) <= c.suspect_radius}
            if new:
                t.suspects |= new
                self._event(f, t, "suspects", f"near the loss point: {' '.join(map(str, sorted(t.suspects)))}")

        # 3. confirmations (anywhere), strongest first
        for tr, r in sorted(present.items(), key=lambda kv: -kv[1].score):
            if not r.score > c.conf_thr:
                continue
            owner = next((t for t in self.targets if t.state in ACTIVE and t.track_id == tr), None)
            if owner is None:   # a plausible continuation of a LOST (or still revisable UNCERTAIN) target?
                lost = [(cont, t) for t in self.targets if t.state == "LOST" or self._revisable(t, f)
                        for cont in [self._continuation(t, f, {tr: r}, self._owned(exclude=t))] if cont is not None]
                if lost:
                    cont, owner = min(lost, key=lambda ct: ct[0][0])
                    owner.track_id = tr
                    self._move(owner, f, r)
                    self._event(f, owner, "found again", cont[2])
            if owner is None:
                owner = Target(self.next_tid, "CONFIRMED", tr, np.array([r.x1, r.y1, r.x2, r.y2], float), f, f, "")
                owner.hist.append((f, float(r.x), float(r.y)))
                self.next_tid += 1
                self.targets.append(owner)
                self._event(f, owner, "new target", f"tag confirmed on track {tr}: score {r.score:.3f}")
            else:
                if owner.state != "CONFIRMED" or owner.last_conf != f - 1:   # log only the first of a run
                    self._event(f, owner, "confirmed", f"tag on track {tr}: score {r.score:.3f}")
            owner.state, owner.last_conf = "CONFIRMED", f
            self._clear_suspects(owner)

        # 4. hits in context: on an UNCERTAIN target's own track, then on its suspects
        for t in self.targets:
            if t.state == "UNCERTAIN" and t.track_id in present:
                s = float(present[t.track_id].score)
                if s >= c.hit_thr:
                    t.state = "CONFIRMED"     # green again, but not a confirmation: last_conf unchanged
                    self._event(f, t, "confirmed by hit", f"hit on followed track {t.track_id}: score {s:.3f}")
                    if not c.suspects_until_confirmed:
                        self._clear_suspects(t)
            if not t.suspects or (t.state == "CONFIRMED" and not c.suspects_until_confirmed):
                continue
            others = self._owned(exclude=t)
            hits = [(float(present[tr].score), tr) for tr in t.suspects
                    if tr in present and tr not in others and present[tr].score >= c.hit_thr and self._moving(tr)]
            if hits:
                s, tr = max(hits)
                if t.track_id in present:
                    t.suspects.add(t.track_id)       # the one we leave stays a suspect
                t.track_id, t.state = tr, "CONFIRMED"
                self._move(t, f, present[tr])
                self._event(f, t, "suspect hit", f"hit on suspect track {tr}: score {s:.3f} (moved "
                                                 f"{self._moved(tr):.0f} px in {c.move_frames} frames)")
                if c.suspects_until_confirmed:
                    t.suspects.discard(tr)
                else:
                    self._clear_suspects(t)

        # 5. output rows (LOST targets at their last position). Green made by a hit while the
        #    suspect window is open is reported UNCERTAIN: a later suspect hit can still move it.
        return [dict(frame=f, target=t.tid, track_id=t.track_id,
                     state="UNCERTAIN" if t.state == "CONFIRMED" and t.suspect_since is not None else t.state,
                     x=round(t.xy[0], 1), y=round(t.xy[1], 1), x1=t.box[0], y1=t.box[1], x2=t.box[2], y2=t.box[3],
                     last_conf=t.last_conf, reason=t.reason,
                     suspects=" ".join(str(s) for s in sorted(t.suspects) if s in present)) for t in self.targets]

    def _clear_suspects(self, t: Target) -> None:
        t.suspects, t.suspect_since = set(), None

    def _moved(self, tr: int) -> float:
        p = np.asarray(self.recent.get(tr, []), float)
        return float(np.hypot(np.ptp(p[:, 0]), np.ptp(p[:, 1]))) if len(p) > 1 else 0.0

    def _moving(self, tr: int) -> bool:
        return self._moved(tr) >= self.cfg.suspect_move_px

    def _move(self, t: Target, f: int, r: pd.Series) -> None:
        if t.track_id != t.counted_track:
            t.on_track, t.counted_track = 0, t.track_id
        t.on_track += 1
        t.box = np.array([r.x1, r.y1, r.x2, r.y2], float)
        t.last_frame = f
        t.hist.append((f, float(r.x), float(r.y)))


def run(tracks: pd.DataFrame, n_frames: int, cfg: FollowConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Follow over frames 0..n_frames-1. tracks: per (frame, track_id) box + score.
    Returns (targets: one row per target per frame, events)."""
    fol = Follower(cfg)
    by_frame = {f: g for f, g in tracks.groupby("frame")}
    empty = tracks.iloc[:0]
    rows = []
    for f in range(n_frames):
        rows += fol.step(f, by_frame.get(f, empty))
    cols = ["frame", "target", "state", "track_id", "x", "y", "x1", "y1", "x2", "y2", "last_conf", "reason", "suspects"]
    return pd.DataFrame(rows, columns=cols), pd.DataFrame(fol.events, columns=["frame", "target", "event", "track_id", "reason"])


def fill_gaps(targets: pd.DataFrame, cfg: FollowConfig) -> pd.DataFrame:
    """End-of-video fill (saved CSV only): a LOST stretch of at most fill_max frames between two
    active pieces of the same target becomes FILLED, with source 'held' (slow before the loss and
    found again within still_dist: stayed in place) or 'interp' (straight line between the boxes)."""
    out = targets.copy()
    out["source"] = np.where(out.state.isin(ACTIVE), "live", "")
    for _, g in out.groupby("target"):
        idx = g.index.to_numpy()
        lost = (g.state == "LOST").to_numpy()
        k = 0
        while k < len(idx):
            if not lost[k]:
                k += 1
                continue
            e = k
            while e + 1 < len(idx) and lost[e + 1]:
                e += 1
            a, b = k - 1, e + 1
            if a >= 0 and b < len(idx) and e - k + 1 <= cfg.fill_max:
                A, B = out.loc[idx[a]], out.loc[idx[b]]
                before = g.iloc[max(0, a - cfg.speed_frames):a + 1]
                speed = float(np.hypot(before.x.diff(), before.y.diff()).mean()) if len(before) > 1 else 0.0
                held = speed < cfg.still_speed and np.hypot(B.x - A.x, B.y - A.y) <= cfg.still_dist
                fa, fb = A.frame, B.frame
                for j in range(k, e + 1):
                    w = 0.0 if held else (out.at[idx[j], "frame"] - fa) / (fb - fa)
                    for col in ("x1", "y1", "x2", "y2"):
                        out.at[idx[j], col] = A[col] + w * (B[col] - A[col])
                    out.at[idx[j], "x"] = round((out.at[idx[j], "x1"] + out.at[idx[j], "x2"]) / 2, 1)
                    out.at[idx[j], "y"] = round((out.at[idx[j], "y1"] + out.at[idx[j], "y2"]) / 2, 1)
                    out.at[idx[j], "state"] = "FILLED"
                    out.at[idx[j], "source"] = "held" if held else "interp"
                    out.at[idx[j], "last_conf"] = A.last_conf
            k = e + 1
    return out


def main_output(targets: pd.DataFrame, n_frames: int) -> pd.DataFrame:
    """One row per frame: the target confirmed most recently (decision 1), LOST ones included:
    while it is LOST the output says "not present" rather than jumping to an older target.
    Columns frame, staff_present, x, y + detail (target, state, track_id, source)."""
    best = targets.sort_values(["frame", "last_conf", "target"], ascending=[True, False, True]).drop_duplicates("frame")
    out = pd.DataFrame({"frame": np.arange(n_frames)})
    cols = ["frame", "x", "y", "target", "state", "track_id"] + (["source"] if "source" in best else [])
    out = out.merge(best[cols], on="frame", how="left")
    shown = out.state.isin(ACTIVE + ("FILLED",))
    out.loc[~shown, ["x", "y"]] = np.nan
    out.insert(1, "staff_present", shown.astype(int))
    return out


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """(first, last) index of each run of True."""
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return []
    splits = np.flatnonzero(np.diff(idx) > 1)
    return list(zip(np.r_[idx[0], idx[splits + 1]].tolist(), np.r_[idx[splits], idx[-1]].tolist()))


def score(main: pd.DataFrame, key: pd.DataFrame, events: pd.DataFrame, right_px: float = 25.0) -> dict:
    """The three agreed numbers against the answer key (key rows with staff_present == 1)."""
    k = key.set_index("frame")
    m = main.set_index("frame")
    kp = (k.staff_present == 1).reindex(m.index, fill_value=False).to_numpy()
    said = (m.staff_present == 1).to_numpy()
    d = np.hypot(m.x - k.x.reindex(m.index), m.y - k.y.reindex(m.index)).to_numpy()
    right = said & kp & (d <= right_px)
    wrong = said & ~right
    conf = events[events.event.isin(["new target", "confirmed", "found again", "confirmed by hit"])]
    first = int(conf.frame.min()) if len(conf) else None
    after = np.arange(len(m)) >= (first if first is not None else len(m))
    stretches = []
    for s, e in runs(wrong):
        seg = m.iloc[s:e + 1]
        stretches.append(dict(start=s, end=e, frames=e - s + 1, targets=sorted(seg.target.dropna().astype(int).unique().tolist()),
                              tracks=sorted(seg.track_id.dropna().astype(int).unique().tolist()),
                              states=sorted(seg.state.dropna().unique().tolist()),
                              median_px=None if np.all(np.isnan(d[s:e + 1])) else round(float(np.nanmedian(d[s:e + 1])), 1)))
    by_state = {}
    for st in sorted(m.state.dropna().unique()):
        sel = (m.state == st).to_numpy()
        by_state[st] = dict(frames=int(sel.sum()), right=int((sel & right).sum()))
    return dict(
        first_confirmation=first,
        said_staff=int(said.sum()), right_when_said=int(right.sum()),
        precision=round(float(right.sum() / max(said.sum(), 1)), 4),
        frames_after_first=int((after & kp).sum()), covered_right_after_first=int((after & kp & right).sum()),
        coverage_after_first=round(float((after & kp & right).sum() / max((after & kp).sum(), 1)), 4),
        right_all_frames=round(float(right.sum() / max(kp.sum(), 1)), 4),
        by_state=by_state, wrong_stretches=stretches)
