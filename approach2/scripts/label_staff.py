"""Step B of the answer key: click tool to label the staff member, helped by ByteTrack tracks.

You click the staff member once; the tool follows that track. When the track ends it
suggests the track that most likely continues it; you accept (Enter), pick another
candidate, or click the right person. Short gaps (<= --max-gap-s) are filled by straight-line
interpolation; longer gaps stay "unlabelled" until you resolve them. x, y = box centre.
The tool also stops at long filled gaps (>= --check-interp-s), even inside one track, so you
can check that the straight line really followed the person.

Range mode (--from/--to in seconds, or --from-frame/--to-frame): you can only move inside the
range, and everything you do only changes frames inside it; labels outside stay exactly as
they were. Use it to fix one problem at a time.

Needs tracks from scripts/track_rectified.py.

Keys
  mouse click   this is the staff member (picks the track box under the click; a click
                outside every box is stored as a manual point with the size of the nearest
                track box). To fix a bad fill: step through it and click the person every few
                frames; the fill then runs between your clicks.
  Enter         accept the shown suggestion, or: this filled gap looks right
  1 / 2 / 3     show suggestion 1 / 2 / 3
  p             play the gap (up to the shown suggestion, or the filled gap) with its yellow fill
  g             go to the next issue: an unlabelled frame or a long filled gap
  n             staff NOT visible from this frame (until the next label)
  x             the labelled track is NOT the staff from this frame on (fix an ID swap)
  u             undo the last action
  a / d         previous / next frame            z / c   back / forward 1 second
  w             save                             q / Esc   save and quit

Box colours: grey = other tracks, green = staff (track), yellow = staff (interpolated),
magenta = staff (manual click), cyan = suggestion, red cross = last known staff position.

Each run of the tool is session N (next free number). Every action is saved to
ground_truth/staff_labels.json with its session (and range); the answer key is saved as
ground_truth/ground_truth_N.csv (kept for reference) and ground_truth/ground_truth.csv (latest).
On quit a review video of the session's range is written to outputs/staff_labels_review_N.mp4
(skip with --no-review). Delete staff_labels.json to start over.

Usage (from approach2/):
    ../.venv/bin/python scripts/label_staff.py
    ../.venv/bin/python scripts/label_staff.py --start 275 --max-gap-s 2.5 --scale 1.2
    ../.venv/bin/python scripts/label_staff.py --from 32 --to 34        # fix one range only
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from staffid2.common import ROOT, SAMPLE_VIDEO, setup_logging  # noqa: E402
from staffid2.drawing import CYAN, GREY, RED, SOURCE_COLOR, WHITE, YELLOW, draw_box, table_video  # noqa: E402
from staffid2.labeling import build_labels, jumps, manual_box, next_issue, suggest, track_at  # noqa: E402

WIN = "label staff"
KEY_HELP = "click=staff  Enter=accept/ok  1-3=candidate  p=play gap  g=next issue  n=not visible  x=cut  u=undo  a/d z/c=move  w=save  q=quit"


def next_free(base: Path) -> Path:
    """base_1.ext, base_2.ext, ...: the first that does not exist yet."""
    n = 1
    while (p := base.with_name(f"{base.stem}_{n}{base.suffix}")).exists():
        n += 1
    return p


class FrameReader:
    """Random access to video frames; sequential reads avoid a seek."""

    def __init__(self, path: str):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"cannot open video: {path}")
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
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


class Labeller:
    def __init__(self, args: argparse.Namespace, log):
        self.a, self.log = args, log
        self.reader = FrameReader(args.video)
        self.tracks = pd.read_csv(args.tracks)
        self.by_frame = {f: g for f, g in self.tracks.groupby("frame")}
        self.max_gap = int(round(args.max_gap_s * self.reader.fps))
        self.state_path = Path(args.out_dir) / "staff_labels.json"
        self.actions: list[dict] = []
        if self.state_path.exists():
            self.actions = json.loads(self.state_path.read_text())["actions"]
            log.info("resumed %d actions from %s", len(self.actions), self.state_path)
        self.labels = build_labels(self.actions, self.tracks, self.reader.n, self.max_gap)
        self.min_interp = max(1, int(round(args.check_interp_s * self.reader.fps)))
        # range mode: only frames lo..hi can be viewed or changed
        fps, last = self.reader.fps, self.reader.n - 1
        self.lo = args.from_frame if args.from_frame is not None else (
            int(round(args.from_s * fps)) if args.from_s is not None else 0)
        self.hi = args.to_frame if args.to_frame is not None else (
            int(round(args.to_s * fps)) if args.to_s is not None else last)
        self.lo, self.hi = max(0, self.lo), min(last, self.hi)
        if self.lo > self.hi:
            raise ValueError(f"empty range: frames {self.lo}-{self.hi}")
        self.scope = [self.lo, self.hi] if (self.lo, self.hi) != (0, last) else None
        self.session = int(next_free(Path(args.out_dir) / "ground_truth.csv").stem.rsplit("_", 1)[1])
        log.info("session %d, frames %d-%d%s", self.session, self.lo, self.hi, " (range mode)" if self.scope else "")
        self.cur = self.lo if self.scope else int(np.clip(args.start, self.lo, self.hi))
        self.cands: list[dict] = []
        self.shown = 0
        self.interp_run: tuple[int, int] | None = None
        self.msg = ("click the staff member to start" if not self.actions
                    else f"session {self.session}: press g for the next issue")
        self.click: tuple[float, float] | None = None

    # ---------------------------------------------------------------- actions
    def add(self, action: dict) -> None:
        action["session"] = self.session
        if self.scope:
            action["scope"] = self.scope
        self.actions.append(action)
        self.log.info("action %d: %s", len(self.actions), action)
        self.labels = build_labels(self.actions, self.tracks, self.reader.n, self.max_gap)
        self.save(quiet=True)

    def undo(self) -> None:
        """Undo the last action of THIS session only (earlier sessions are never touched)."""
        if not self.actions or self.actions[-1].get("session") != self.session:
            self.msg = "nothing to undo in this session"
            return
        if self.actions:
            self.log.info("undo: %s", self.actions.pop())
            self.labels = build_labels(self.actions, self.tracks, self.reader.n, self.max_gap)
            self.save(quiet=True)
            self.msg = "undone"

    def goto_issue(self, start: int) -> None:
        self.cands, self.interp_run = [], None
        issue = next_issue(self.labels, self.actions, max(start, self.lo), self.hi, self.min_interp)
        if issue is None:
            self.msg = "nothing left to check in this range: q to save and quit"
            return
        kind, g, e = issue
        if kind == "interp":
            self.interp_run, self.cur = (g, e), g
            self.msg = (f"filled gap {g}-{e} ({e - g + 1} frames): p = play, Enter = looks right, "
                        f"or step through (d/c) and click the person / n")
            return
        self.cands = [c for c in suggest(self.labels, self.tracks, g, self.max_gap, self.a.base_dist,
                                         self.a.max_speed, gap_weight=self.a.gap_weight) if c["frame"] <= self.hi]
        self.shown = 0
        if self.cands:
            self.cur = self.cands[0]["frame"]
            self.msg = f"track lost at frame {g}: suggestion 1 of {len(self.cands)} (Enter = accept, p = play gap)"
        else:
            self.cur = g
            self.msg = f"track lost at frame {g}: no suggestion; move on and click the staff, or n = not visible"

    def on_click(self, x: float, y: float) -> None:
        tid = track_at(self.tracks, self.cur, x, y)
        if tid is not None:
            self.add({"type": "track", "frame": self.cur, "track_id": tid})
            self.goto_issue(self.cur)
        else:
            x1, y1, x2, y2 = manual_box(self.labels, self.cur, x, y)
            self.add({"type": "manual", "frame": self.cur, "x": round(x, 1), "y": round(y, 1),
                      "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)})
            self.cands, self.interp_run = [], None
            self.msg = f"point added at frame {self.cur}: step on (d/c) and click again, or g = next issue"

    def accept(self) -> None:
        if self.interp_run:
            s, e = self.interp_run
            self.add({"type": "ok", "frame": s, "to": e})
            self.goto_issue(e + 1)
            return
        if not self.cands:
            self.msg = "nothing to accept"
            return
        c = self.cands[self.shown]
        self.add({"type": "track", "frame": c["frame"], "track_id": c["track_id"]})
        self.goto_issue(c["frame"])

    # ---------------------------------------------------------------- drawing
    def last_known(self, before: int) -> pd.Series | None:
        lab = self.labels[(self.labels.frame < before) & self.labels.source.isin(("track", "manual"))]
        return lab.iloc[-1] if len(lab) else None

    def render(self, i: int, interp_line: tuple | None = None) -> np.ndarray:
        img = self.reader.get(i).copy()
        for r in self.by_frame.get(i, pd.DataFrame()).itertuples():
            draw_box(img, (r.x1, r.y1, r.x2, r.y2), GREY, str(r.track_id), thickness=1, font_scale=0.4, text_inside=True)
        lab = self.labels.loc[i]
        if lab.source in SOURCE_COLOR:
            c = SOURCE_COLOR[lab.source]
            draw_box(img, (lab.x1, lab.y1, lab.x2, lab.y2), c, thickness=3)
            cv2.circle(img, (int(lab.x), int(lab.y)), 4, c, -1)
        if self.cands and i == self.cands[self.shown]["frame"]:
            c = self.cands[self.shown]
            r = self.tracks[(self.tracks.frame == i) & (self.tracks.track_id == c["track_id"])].iloc[0]
            draw_box(img, (r.x1, r.y1, r.x2, r.y2), CYAN, thickness=3)
            last = self.last_known(i)
            if last is not None:
                cv2.drawMarker(img, (int(last.x), int(last.y)), RED, cv2.MARKER_CROSS, 22, 3)
                cv2.line(img, (int(last.x), int(last.y)), (int(c["x"]), int(c["y"])), CYAN, 1)
        if interp_line is not None:
            cv2.circle(img, (int(interp_line[0]), int(interp_line[1])), 7, YELLOW, 2)
        return self.frame_with_bars(img, i)

    def frame_with_bars(self, img: np.ndarray, i: int) -> np.ndarray:
        lab = self.labels.loc[i]
        rng = f"  [session {self.session}, frames {self.lo}-{self.hi}]" if self.scope else f"  [session {self.session}]"
        status = f"frame {i}/{self.reader.n - 1}{rng}  staff: {lab.source}"
        if lab.source == "track":
            status += f" (track {lab.track_id})"
        if self.cands:
            status += "   candidates: " + "  ".join(
                f"{k + 1}{'*' if k == self.shown else ''}: track {c['track_id']} @ {c['frame']} "
                f"(+{c['gap']}f, {c['dist']:.0f}px)" for k, c in enumerate(self.cands))
        w = img.shape[1]
        top = np.zeros((44, w, 3), np.uint8)
        cv2.putText(top, status[:150], (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)
        cv2.putText(top, self.msg[:150], (6, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.45, CYAN, 1)
        bottom = np.zeros((22, w, 3), np.uint8)
        cv2.putText(bottom, KEY_HELP[:170], (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, GREY, 1)
        out = np.vstack([top, img, bottom])
        if self.a.scale != 1.0:
            out = cv2.resize(out, None, fx=self.a.scale, fy=self.a.scale)
        return out

    def play_gap(self) -> None:
        """Play from the last known position to the shown candidate, with the straight-line
        interpolation the tool would use drawn as a yellow circle."""
        delay = max(1, int(1000 / (self.reader.fps * self.a.play_speed)))
        if self.interp_run:  # the yellow box is the fill itself
            s, e = self.interp_run
            for i in range(max(s - 5, self.lo), min(e + 5, self.hi) + 1):
                cv2.imshow(WIN, self.render(i))
                if cv2.waitKey(delay) & 0xFF in (27, ord("p")):
                    break
            self.cur = s
            self.msg = f"played {s}-{e}: did the yellow box stay on the staff? Enter = yes, else click/n to fix"
            return
        if not self.cands:
            self.msg = "no suggestion: nothing to play"
            return
        c = self.cands[self.shown]
        last = self.last_known(c["frame"])
        if last is None:
            return
        f0, f1 = max(int(last.frame), self.lo), c["frame"]
        for i in range(f0, f1 + 1):
            t = (i - f0) / max(f1 - f0, 1)
            p = (last.x + t * (c["x"] - last.x), last.y + t * (c["y"] - last.y))
            cv2.imshow(WIN, self.render(i, interp_line=p))
            if cv2.waitKey(delay) & 0xFF in (27, ord("p")):
                break
        self.cur = f1
        self.msg = f"played frames {f0}-{f1}: did the yellow circle stay on the staff? Enter = accept"

    # ---------------------------------------------------------------- saving
    def save(self, quiet: bool = False) -> None:
        out = Path(self.a.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        meta = dict(video=self.a.video, tracks=self.a.tracks, max_gap_s=self.a.max_gap_s, max_gap_frames=self.max_gap,
                    saved=time.strftime("%Y-%m-%d %H:%M:%S"), labelled_by="user (label_staff.py)",
                    actions=self.actions)
        self.state_path.write_text(json.dumps(meta, indent=1))
        self.labels.round(1).to_csv(out / "ground_truth.csv", index=False)
        self.labels.round(1).to_csv(out / f"ground_truth_{self.session}.csv", index=False)
        if not quiet:
            self.log.info("saved %s, %s and ground_truth_%d.csv", self.state_path, out / "ground_truth.csv",
                          self.session)
            self.msg = "saved"

    def summary(self) -> None:
        lab = self.labels
        self.log.info("label sources: %s", lab.source.value_counts().to_dict())
        for name in ("unlabelled", "interp", "absent"):
            sel = lab[lab.source == name].frame
            runs = [(int(r.min()), int(r.max())) for _, r in sel.groupby((sel.diff() != 1).cumsum())]
            self.log.info("%s ranges (%d): %s", name, len(runs), runs)
        j = jumps(lab, self.a.jump_px)
        self.log.info("centre jumps > %.0f px between consecutive frames (check for ID swaps): %s",
                      self.a.jump_px, j[["frame", "jump_px"]].values.tolist())

    def review_video(self) -> None:
        path = Path(self.a.review_out) if self.a.review_out else (
            ROOT / "outputs" / f"staff_labels_review_{self.session}.mp4")
        source = lambda i, rows: rows.source.iloc[0] if len(rows) else "unlabelled"  # noqa: E731
        table_video(self.a.video, self.labels, path, color=lambda r: SOURCE_COLOR.get(r.source, WHITE),
                    header=lambda i, rows: f"frame {i}  staff: {source(i, rows)}",
                    header_color=lambda i, rows: SOURCE_COLOR.get(source(i, rows), WHITE), thickness=3,
                    start=self.lo, end=self.hi)
        self.log.info("review video: %s", path)

    # ---------------------------------------------------------------- main loop
    def run(self) -> None:
        cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)

        def mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.click = (x / self.a.scale, y / self.a.scale - 44)  # undo scale and top bar

        cv2.setMouseCallback(WIN, mouse)
        while True:
            cv2.imshow(WIN, self.render(self.cur))
            key = cv2.waitKey(30) & 0xFF
            if self.click is not None:
                x, y = self.click
                self.click = None
                if 0 <= y < self.reader.get(self.cur).shape[0]:
                    self.on_click(x, y)
                continue
            if key == 255:
                continue
            ch = chr(key) if key < 128 else ""
            if key in (13, 10):
                self.accept()
            elif ch in "123" and ch:
                k = int(ch) - 1
                if k < len(self.cands):
                    self.shown, self.cur = k, self.cands[k]["frame"]
                    self.msg = f"showing suggestion {k + 1}: Enter = accept, p = play gap"
            elif ch == "p":
                self.play_gap()
            elif ch == "g":
                self.goto_issue(self.cur)
            elif ch == "n":
                self.add({"type": "absent", "frame": self.cur})
                self.cands, self.interp_run = [], None
                self.msg = f"staff not visible from {self.cur}: move on and click when they reappear"
            elif ch == "x":
                lab = self.labels.loc[self.cur]
                if lab.source == "track":
                    self.add({"type": "cut", "frame": self.cur, "track_id": int(lab.track_id)})
                    self.goto_issue(self.cur)
                else:
                    self.msg = "x only works on a frame labelled from a track"
            elif ch == "u":
                self.undo()
            elif ch in "adzc" and ch:
                step = {"a": -1, "d": 1, "z": -int(self.reader.fps), "c": int(self.reader.fps)}[ch]
                self.cur = int(np.clip(self.cur + step, self.lo, self.hi))
                self.cands, self.interp_run, self.msg = [], None, ""
            elif ch == "w":
                self.save()
            elif ch == "q" or key == 27:
                break
        cv2.destroyAllWindows()
        self.save()
        self.summary()
        if not self.a.no_review:
            self.review_video()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", default=str(SAMPLE_VIDEO))
    ap.add_argument("--tracks", default=str(ROOT / "data" / "track_rectified" / "tracks.csv"))
    ap.add_argument("--start", type=int, default=0, help="first frame to show (without a range)")
    ap.add_argument("--from", dest="from_s", type=float, default=None, help="range start in seconds")
    ap.add_argument("--to", dest="to_s", type=float, default=None, help="range end in seconds")
    ap.add_argument("--from-frame", type=int, default=None, help="range start frame (overrides --from)")
    ap.add_argument("--to-frame", type=int, default=None, help="range end frame (overrides --to)")
    ap.add_argument("--check-interp-s", type=float, default=0.4,
                    help="g also stops at filled gaps at least this long (seconds)")
    ap.add_argument("--max-gap-s", type=float, default=2.5, help="gaps up to this long are interpolated")
    ap.add_argument("--base-dist", type=float, default=100.0, help="suggestion: allowed distance at gap 0 (px)")
    ap.add_argument("--max-speed", type=float, default=30.0, help="suggestion: max plausible speed (px/frame)")
    ap.add_argument("--gap-weight", type=float, default=3.0, help="suggestion: preference for early re-appearance")
    ap.add_argument("--jump-px", type=float, default=60.0, help="report frame-to-frame jumps larger than this")
    ap.add_argument("--play-speed", type=float, default=1.0, help="gap playback speed (1 = real time)")
    ap.add_argument("--scale", type=float, default=1.0, help="window scale")
    ap.add_argument("--out-dir", default=str(ROOT / "ground_truth"))
    ap.add_argument("--review-out", default=None, help="default: outputs/staff_labels_review_N.mp4 (next free N)")
    ap.add_argument("--no-review", action="store_true")
    args = ap.parse_args()

    log = setup_logging("label_staff")
    log.info("params: %s", vars(args))
    Labeller(args, log).run()


if __name__ == "__main__":
    main()
