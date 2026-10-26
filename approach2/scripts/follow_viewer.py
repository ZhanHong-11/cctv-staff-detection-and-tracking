"""Follow viewer (user tool, UNDER REVIEW): step 4 following of the staff member, frame by frame.

Reads what follow_staff.py decided (outputs/follow/) and the tag scores; it
decides nothing itself. Two views (l toggles):
  live   what the follower knew at that frame (LOST targets as red crosses, nothing filled)
  saved  after the end fill (short LOST stretches filled: interp / held)

Boxes
  green, thick           CONFIRMED target (tag seen on this track)
  light green, dashed    UNCERTAIN target (following a continuation, no tag on it yet)
  white, dotted          FILLED (saved view only)
  red cross              LOST target at its last position, while it is searched for
  orange                 a hit (>= hit_thr) on this box now or in the last focus_time frames
  grey                   everyone else
  yellow                 suspect: a person near where a target was lost (old tracks too); a hit
                         on a suspect moves the target there (4.5 case 1)
  "STAFF"                the target the main CSV reports in this frame
  thin cyan              the answer key's staff box (g toggles it; for checking only)

Bottom strips: main output per frame (green confirmed / light green uncertain / white filled /
red main target lost / dark nothing), then right vs the answer key (green right, red wrong
person, grey staff not reported). Yellow ticks = confirmations. Click a strip to jump.

Keys
  a / d   previous / next frame       z / c   back / forward 1 second
  n / b   next / previous event (new target, lost, continuation, switch, revived, confirmed ...)
  w / e   next / previous wrong stretch            l   live / saved view
  p       play (any key stops)                     g   answer-key box on / off
  v       score text on all boxes / only hits      q / Esc quit

Usage (from approach2/):
    ../.venv/bin/python scripts/follow_viewer.py
    ../.venv/bin/python scripts/follow_viewer.py --start 860 --scale 0.85
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from staffid2.common import ROOT, ANSWER_KEY, SAMPLE_VIDEO, TAG_SCORES, setup_logging  # noqa: E402
from staffid2.drawing import CYAN, GREEN, GREY, RED, WHITE, YELLOW  # noqa: E402

WIN = "follow viewer"
TOP, STRIP, BOTTOM = 44, 22, 36
RIGHT_W = 440
ORANGE, LIGHT_GREEN, DARK = (0, 150, 255), (150, 240, 150), (60, 60, 60)
FONT = cv2.FONT_HERSHEY_SIMPLEX
KEY_HELP = ("a/d=frame  z/c=1 s  n/b=event  w/e=wrong stretch  l=live/saved  p=play  g=key box  "
            "v=texts  q=quit  click strips=jump")
EVENT_COL = {"new target": GREEN, "confirmed": GREEN, "confirmed by hit": GREEN, "found again": GREEN,
             "lost": RED, "dropped": RED, "continuation": LIGHT_GREEN, "switched": LIGHT_GREEN, "revived": LIGHT_GREEN,
             "suspects": YELLOW, "suspect hit": YELLOW}


class FrameReader:
    """Random access to video frames; sequential reads avoid a seek (copied from tag_score_viewer.py)."""

    def __init__(self, path: str):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"cannot open video: {path}")
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.pos = -1
        self.cache: dict[int, np.ndarray] = {}

    def get(self, i: int) -> np.ndarray:
        if i in self.cache:
            return self.cache[i]
        if i != self.pos + 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = self.cap.read()
        if not ok:
            raise ValueError(f"cannot read frame {i}")
        self.pos = i
        if len(self.cache) > 200:
            self.cache.pop(next(iter(self.cache)))
        self.cache[i] = f
        return f


def dashed_rect(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, col, th: int, dash: int, gap: int) -> None:
    for (a, b) in (((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)), ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))):
        n = max(1, int(np.hypot(b[0] - a[0], b[1] - a[1]) // (dash + gap)))
        for k in range(n + 1):
            t0, t1 = k * (dash + gap), k * (dash + gap) + dash
            L = max(np.hypot(b[0] - a[0], b[1] - a[1]), 1)
            if t0 >= L:
                break
            p = lambda t: (int(a[0] + (b[0] - a[0]) * min(t, L) / L), int(a[1] + (b[1] - a[1]) * min(t, L) / L))
            cv2.line(img, p(t0), p(t1), col, th)


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return []
    splits = np.flatnonzero(np.diff(idx) > 1)
    return list(zip(np.r_[idx[0], idx[splits + 1]].tolist(), np.r_[idx[splits], idx[-1]].tolist()))


class FollowViewer:
    def __init__(self, args: argparse.Namespace, log):
        self.a, self.log = args, log
        self.reader = FrameReader(args.video)
        n = self.reader.n
        d = Path(args.follow)
        cfg = json.loads((d / "score.json").read_text())["settings"]
        self.hit_thr, self.conf_thr, self.focus = cfg["hit_thr"], cfg["conf_thr"], cfg["focus_time"]
        scores = pd.read_csv(args.scores)
        self.boxes = {f: g for f, g in scores.groupby("frame")}
        self.hits = {t: np.sort(g.frame.to_numpy()) for t, g in scores[scores.score >= self.hit_thr].groupby("track_id")}
        self.events = pd.read_csv(d / "events.csv")
        key = pd.read_csv(args.answer_key)
        self.key = key[(key.staff_present == 1) & key.x1.notna()].set_index("frame")
        self.views = {}
        for name, tfile, mfile in (("live", "targets_live.csv", "main_live_detail.csv"), ("saved", "targets.csv", "main_detail.csv")):
            t = pd.read_csv(d / tfile)
            m = pd.read_csv(d / mfile).set_index("frame").reindex(range(n))
            kx, ky = self.key.x.reindex(range(n)), self.key.y.reindex(range(n))
            said = (m.staff_present == 1).to_numpy()
            right = said & (np.hypot(m.x - kx, m.y - ky).to_numpy() <= args.right_px)
            verdict = np.where(right, 1, np.where(said, 2, np.where(kx.notna().to_numpy(), 3, 0)))  # 1 right 2 wrong 3 missed
            self.views[name] = dict(targets={f: g for f, g in t.groupby("frame")}, main=m, verdict=verdict,
                                    wrong=np.array([s for s, _ in runs(verdict == 2)], int))
        self.view = "live"
        self.cur = int(np.clip(args.start, 0, n - 1))
        self.event_frames = np.unique(self.events[~self.events.event.isin(["confirmed", "suspects"])].frame.to_numpy())
        self.conf_frames = np.unique(self.events[self.events.event.isin(["new target", "confirmed", "found again"])].frame)
        self.all_text, self.show_key = False, True
        self.click: tuple[int, int] | None = None
        self.strips: dict[str, np.ndarray] = {}
        self.msg = "live view: what the follower knew at each frame (l = saved view with gaps filled)"
        for name, v in self.views.items():
            log.info("%s: right %d, wrong %d, missed %d frames; wrong stretches start at %s", name,
                     (v["verdict"] == 1).sum(), (v["verdict"] == 2).sum(), (v["verdict"] == 3).sum(), v["wrong"].tolist())

    # ---------------------------------------------------------------- drawing
    def recent_hit(self, tid: int) -> bool:
        h = self.hits.get(tid)
        if h is None:
            return False
        i = np.searchsorted(h, self.cur, side="right")
        return i > 0 and self.cur - h[i - 1] <= self.focus

    def render(self) -> np.ndarray:
        frame = self.reader.get(self.cur).copy()
        v = self.views[self.view]
        if self.show_key and self.cur in self.key.index:
            g = self.key.loc[self.cur]
            cv2.rectangle(frame, (int(g.x1) - 5, int(g.y1) - 5), (int(g.x2) + 5, int(g.y2) + 5), CYAN, 1)
            cv2.putText(frame, "key", (int(g.x1) - 5, int(g.y2) + 17), FONT, 0.4, CYAN, 1)
        tg = v["targets"].get(self.cur)
        suspects = set()
        if tg is not None and "suspects" in tg:
            for sv in tg.suspects.dropna():
                suspects |= {int(x) for x in str(sv).split()}
        for _, r in self.boxes.get(self.cur, pd.DataFrame(columns=["track_id"])).iterrows():
            hit_now, recent = r.score >= self.hit_thr, self.recent_hit(int(r.track_id))
            col = YELLOW if int(r.track_id) in suspects else ORANGE if recent else GREY
            cv2.rectangle(frame, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), col,
                          2 if hit_now or int(r.track_id) in suspects else 1)
            if int(r.track_id) in suspects:
                cv2.putText(frame, f"suspect {int(r.track_id)}", (int(r.x1), int(r.y1) + 14), FONT, 0.4, YELLOW, 1)
            if self.all_text or hit_now:
                cv2.putText(frame, f"{r.score:.3f}", (int(r.x1), int(r.y2) + 13), FONT, 0.4, col, 1)
        m = v["main"].loc[self.cur]
        for _, t in v["targets"].get(self.cur, pd.DataFrame(columns=["state"])).iterrows():
            x1, y1, x2, y2 = (int(t[c]) for c in ("x1", "y1", "x2", "y2"))
            name = f"T{int(t.target)}"
            if t.state == "CONFIRMED":
                cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN, 3)
                col, text = GREEN, f"{name} confirmed  tr {int(t.track_id)}"
            elif t.state == "UNCERTAIN":
                dashed_rect(frame, x1, y1, x2, y2, LIGHT_GREEN, 2, 8, 6)
                col, text = LIGHT_GREEN, f"{name} uncertain  tr {int(t.track_id)}"
            elif t.state == "FILLED":
                dashed_rect(frame, x1, y1, x2, y2, WHITE, 1, 3, 4)
                col, text = WHITE, f"{name} filled ({t.source})"
            else:
                cx, cy = int(t.x), int(t.y)
                cv2.drawMarker(frame, (cx, cy), RED, cv2.MARKER_TILTED_CROSS, 22, 3)
                lost_for = self.cur - int(self.events[(self.events.target == t.target) & (self.events.event == "lost")
                                                      & (self.events.frame <= self.cur)].frame.max())
                cv2.putText(frame, f"{name} lost {lost_for} fr", (cx + 12, cy - 8), FONT, 0.45, RED, 1)
                continue
            is_main = int(t.target) == m.target and m.staff_present == 1
            cv2.putText(frame, ("STAFF  " if is_main else "") + text, (x1, max(y1 - 6, 12)), FONT, 0.5, col,
                        2 if is_main else 1)
        body = np.hstack([frame, self.right_panel(frame.shape[0])])
        return self.with_bars(body)

    def right_panel(self, h: int) -> np.ndarray:
        p = np.zeros((h, RIGHT_W, 3), np.uint8)
        v = self.views[self.view]
        m = v["main"].loc[self.cur]
        verdict = {0: ("(staff not in key)", GREY), 1: ("RIGHT person", GREEN), 2: ("WRONG person", RED),
                   3: ("staff not reported", GREY)}[int(v["verdict"][self.cur])]
        y = 22

        def line(text, col=WHITE, scale=0.45):
            nonlocal y
            if y < h - 4:
                cv2.putText(p, text, (8, y), FONT, scale, col, 1)
            y += 19

        line(f"{self.view.upper()} view", YELLOW, 0.55)
        if m.staff_present == 1:
            line(f"main CSV: staff at ({m.x:.0f}, {m.y:.0f})  T{int(m.target)} {m.state.lower()}", WHITE)
        else:
            line("main CSV: staff_present = 0" + (f"  (T{int(m.target)} lost)" if pd.notna(m.target) else ""), GREY)
        line(f"vs answer key: {verdict[0]}", verdict[1])
        y += 6
        line("targets", YELLOW)
        rows = v["targets"].get(self.cur)
        if rows is None:
            line("  none: searching for a tag (score > %.3f)" % self.conf_thr, GREY)
        else:
            for _, t in rows.iterrows():
                col = {"CONFIRMED": GREEN, "UNCERTAIN": LIGHT_GREEN, "FILLED": WHITE}.get(t.state, RED)
                line(f"  T{int(t.target)} {t.state.lower()}  track {int(t.track_id)}  last conf. {int(t.last_conf)}", col)
                if isinstance(t.get("suspects"), str) and t.suspects:
                    line(f"     suspects: {t.suspects}", YELLOW)
                reason = str(t.reason)
                for k in range(0, len(reason), 52):
                    line("     " + reason[k:k + 52], GREY)
        y += 6
        line("events up to this frame", YELLOW)
        ev = self.events[(self.events.frame <= self.cur) & ~self.events.event.isin(["confirmed", "suspects"])].tail(10)
        for _, e in ev[::-1].iterrows():
            text = f"{int(e.frame):4d} T{int(e.target)} {e.event}: {e.reason}"
            line("  " + text[:56], EVENT_COL.get(e.event, WHITE))
            if len(text) > 56:
                line("        " + text[56:106], GREY)
        return p

    def make_strips(self, w: int) -> np.ndarray:
        n = self.reader.n
        v = self.views[self.view]
        out = np.full((2 * STRIP + 8, w, 3), 25, np.uint8)
        cols = np.minimum((np.arange(n) / max(n - 1, 1) * (w - 1)).astype(int), w - 1)
        main = v["main"]
        for f in range(n):
            st, pres = main.state.iat[f], main.staff_present.iat[f] == 1
            c = (GREEN if st == "CONFIRMED" else LIGHT_GREEN if st == "UNCERTAIN" else WHITE if st == "FILLED"
                 else RED if isinstance(st, str) and not pres else DARK)
            cv2.line(out, (cols[f], 6), (cols[f], 4 + STRIP), c, 1)
            vc = {0: DARK, 1: GREEN, 2: RED, 3: GREY}[int(v["verdict"][f])]
            cv2.line(out, (cols[f], 8 + STRIP), (cols[f], 6 + 2 * STRIP), vc, 1)
        for f in self.conf_frames:
            cv2.line(out, (cols[f], 0), (cols[f], 4), YELLOW, 1)
        cv2.putText(out, "main output", (w - 110, 18), FONT, 0.38, WHITE, 1)
        cv2.putText(out, "vs answer key", (w - 110, 20 + STRIP), FONT, 0.38, WHITE, 1)
        return out

    def with_bars(self, body: np.ndarray) -> np.ndarray:
        w, n = body.shape[1], self.reader.n
        key = f"{self.view}{w}"
        if key not in self.strips:
            self.strips[key] = self.make_strips(w)
        strips = self.strips[key].copy()
        x = int(self.cur / max(n - 1, 1) * (w - 1))
        cv2.line(strips, (x, 0), (x, strips.shape[0]), (255, 0, 255), 2)
        top = np.zeros((TOP, w, 3), np.uint8)
        status = f"frame {self.cur}/{n - 1}  t={self.cur / self.reader.fps:.1f}s  |  {self.view} view  |  "
        cv2.putText(top, status, (6, 17), FONT, 0.45, WHITE, 1)
        x = 6 + cv2.getTextSize(status, FONT, 0.45, 1)[0][0]
        for text, c in (("confirmed  ", GREEN), ("uncertain  ", LIGHT_GREEN), ("lost x  ", RED), ("filled  ", WHITE),
                        ("suspect  ", YELLOW),
                        (f"hit >= {self.hit_thr} (1 s)  ", ORANGE), ("other  ", GREY), ("key", CYAN)):
            cv2.putText(top, text, (x, 17), FONT, 0.45, c, 1)
            x += cv2.getTextSize(text, FONT, 0.45, 1)[0][0]
        cv2.putText(top, self.msg[:170], (6, 37), FONT, 0.45, CYAN, 1)
        bottom = np.zeros((BOTTOM, w, 3), np.uint8)
        cv2.putText(bottom, KEY_HELP, (6, 22), FONT, 0.42, GREY, 1)
        out = np.vstack([top, body, strips, bottom])
        if self.a.scale != 1.0:
            out = cv2.resize(out, None, fx=self.a.scale, fy=self.a.scale, interpolation=cv2.INTER_AREA)
        return out

    # ---------------------------------------------------------------- interaction
    def go(self, f: int) -> None:
        self.cur = int(np.clip(f, 0, self.reader.n - 1))
        self.msg = ""

    def jump(self, frames: np.ndarray, forward: bool, what: str) -> None:
        cand = frames[frames > self.cur] if forward else frames[frames < self.cur]
        if not len(cand):
            self.msg = f"no {what} {'after' if forward else 'before'} this frame"
            return
        self.go(int(cand.min() if forward else cand.max()))
        if what == "event":
            ev = self.events[(self.events.frame == self.cur) & (self.events.event != "confirmed")]
            self.msg = "; ".join(f"T{int(e.target)} {e.event}: {e.reason}" for _, e in ev.iterrows())

    def on_mouse(self, event, x, y, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.click = (int(x / self.a.scale), int(y / self.a.scale))

    def handle_click(self) -> None:
        x, y = self.click
        self.click = None
        top = TOP + self.reader.h
        if top <= y < top + 2 * STRIP + 8:
            self.go(int(round(x / max(self.reader.w + RIGHT_W - 1, 1) * (self.reader.n - 1))))

    def play(self) -> None:
        delay = max(1, int(1000 / (self.reader.fps * self.a.play_speed)))
        while self.cur < self.reader.n - 1:
            self.go(self.cur + 1)
            cv2.imshow(WIN, self.render())
            if cv2.waitKey(delay) & 0xFF != 255:
                break
        self.msg = f"stopped at {self.cur}"

    def run(self) -> None:
        cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WIN, self.on_mouse)
        fps = int(round(self.reader.fps))
        dirty = True
        while True:
            if dirty:
                cv2.imshow(WIN, self.render())
                dirty = False
            key = cv2.waitKey(30) & 0xFF
            if self.click:
                self.handle_click()
                dirty = True
                continue
            if key == 255:
                continue
            dirty = True
            ch = chr(key) if key < 128 else ""
            if ch in ("a", "d", "z", "c"):
                self.go(self.cur + {"a": -1, "d": 1, "z": -fps, "c": fps}[ch])
            elif ch in ("n", "b"):
                self.jump(self.event_frames, ch == "n", "event")
            elif ch in ("w", "e"):
                self.jump(self.views[self.view]["wrong"], ch == "w", "wrong stretch")
            elif ch == "l":
                self.view = "saved" if self.view == "live" else "live"
                self.msg = f"{self.view} view"
            elif ch == "v":
                self.all_text = not self.all_text
            elif ch == "g":
                self.show_key = not self.show_key
            elif ch == "p":
                self.play()
            elif ch == "q" or key == 27:
                break
        cv2.destroyAllWindows()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", default=str(SAMPLE_VIDEO))
    ap.add_argument("--scores", default=str(TAG_SCORES), help="score_tracks.py output (the follower's input)")
    ap.add_argument("--follow", default=str(ROOT / "outputs" / "follow"), help="follow_staff.py output folder")
    ap.add_argument("--answer-key", default=str(ANSWER_KEY))
    ap.add_argument("--right-px", type=float, default=25.0, help="as follow_staff.py's scoring")
    ap.add_argument("--start", type=int, default=380)
    ap.add_argument("--play-speed", type=float, default=1.0)
    ap.add_argument("--scale", type=float, default=1.0, help="window scale")
    ap.add_argument("--snapshot", type=int, nargs="*", help="save these frames as PNGs to the follow folder and exit")
    args = ap.parse_args()

    log = setup_logging("follow_viewer")
    log.info("params: %s", vars(args))
    v = FollowViewer(args, log)
    if args.snapshot:
        for f in args.snapshot:
            v.go(f)
            out = Path(args.follow) / f"viewer_{f}.png"
            cv2.imwrite(str(out), v.render())
            log.info("wrote %s", out)
        return
    v.run()


if __name__ == "__main__":
    main()
