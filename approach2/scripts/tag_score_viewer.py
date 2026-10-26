"""Tag score viewer (user tool): every tracked person box with its NCC tag score, frame by frame.

Left: the whole frame. Every tracked box is coloured by its tag score (score_tracks.py):
  green  = anchor   score > anchor threshold (0.065: higher than any non-staff box in the sample)
  orange = hit      score >= hit threshold (0.037: 3c no_mid, 2 % of other people's crops pass)
  grey   = below    0 <= score < hit
  dark   = negative the no-bars block fits better than the tag
Hits and anchors also get the matched tag spot outlined (small polygon on the chest).
Thin cyan box = the staff member in the answer key (g toggles it).

Right: the selected box (default: the best score in the frame; click a box to follow that
track). Its rectified crop with the chest search area (dashed) and the matched spot, then
what the camera saw at that spot next to the best template and its two "no bars" blocks, with
their NCC values: score = tag NCC - best block NCC (recomputed live as a check).

Bottom strips: per frame, the best score of any box (top) and of the answer key's staff box
(bottom); bar height = score, colour = band. Cyan ticks = the user's tag sightings. Click to jump.

Keys
  a / d   previous / next frame       z / c   back / forward 1 second
  n / b   next / previous frame with an anchor      . / ,   next / previous frame with a hit
  tab     select the next box (by score)            f       free selection (best box again)
  v       score text on all boxes / only hits       g       answer-key staff box on / off
  p       play (any key stops)                      q / Esc quit
  click a box: follow that track    click the strips: jump to that frame

Usage (from approach2/):
    ../.venv/bin/python scripts/tag_score_viewer.py
    ../.venv/bin/python scripts/tag_score_viewer.py --start 380 --scale 0.85
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from staffid2.common import (ANSWER_KEY, BRIEF_PDF, SAMPLE_VIDEO, TAG_SCORES, TAG_SIGHTINGS,  # noqa: E402
                                    setup_logging)
from staffid2.ncc import BankConfig, MatchConfig, build_bank  # noqa: E402
from staffid2.tag_source import draw_tag, find_tag, measure_layout, pdf_jpegs, to_albedo  # noqa: E402
from staffid2.drawing import CYAN, GREEN, GREY, RED, WHITE, YELLOW  # noqa: E402
from staffid2.person_crop import Lens, person_view, to_original_points  # noqa: E402

WIN = "tag score viewer"
TOP, STRIP, BOTTOM = 44, 26, 36
RIGHT_W = 420
ORANGE, DARK = (0, 150, 255), (90, 90, 90)
EDGE_PTS = 5
KEY_HELP = ("a/d=frame  z/c=1 s  n/b=anchor frame  . ,=hit frame  tab=next box  f=free  v=texts  "
            "g=staff box  p=play  q=quit  click box=follow track  click strips=jump")


class FrameReader:
    """Random access to video frames; sequential reads avoid a seek (copied from label_tag.py)."""

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


def fit(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """img scaled to fit w x h (aspect kept, nearest so pixels stay visible), padded black."""
    s = min(w / img.shape[1], h / img.shape[0])
    small = cv2.resize(img, (max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s))),
                       interpolation=cv2.INTER_NEAREST if s >= 1 else cv2.INTER_AREA)
    out = np.zeros((h, w, 3), np.uint8)
    out[:small.shape[0], :small.shape[1]] = small
    return out


def rect_outline(x0: float, y0: float, w: int, h: int) -> np.ndarray:
    """Points around a crop window (pixel-edge coords) in remap index coords (pixel centre = integer)."""
    t = np.linspace(0, 1, EDGE_PTS)
    x1, y1, x2, y2 = x0 - 0.5, y0 - 0.5, x0 + w - 0.5, y0 + h - 0.5
    xs = np.concatenate([x1 + (x2 - x1) * t, np.full_like(t, x2), x2 - (x2 - x1) * t, np.full_like(t, x1)])
    ys = np.concatenate([np.full_like(t, y1), y1 + (y2 - y1) * t, np.full_like(t, y2), y2 - (y2 - y1) * t])
    return np.stack([xs, ys], axis=1)


def to_gray_bgr(a: np.ndarray, stretch: bool) -> np.ndarray:
    a = a.astype(np.float32)
    if stretch and a.max() > a.min():
        a = (a - a.min()) / (a.max() - a.min()) * 255
    return cv2.cvtColor(np.clip(a, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


class ScoreViewer:
    def __init__(self, args: argparse.Namespace, log):
        self.a, self.log = args, log
        self.reader = FrameReader(args.video)
        self.lens = Lens.for_frame(self.reader.w, self.reader.h, args.fov_deg)
        self.scores = pd.read_csv(args.scores)
        self.by_frame = {f: g.sort_values("score", ascending=False) for f, g in self.scores.groupby("frame")}
        key = pd.read_csv(args.answer_key)
        self.key = key[(key.staff_present == 1) & key.x1.notna()].set_index("frame")
        self.sightings = pd.read_csv(args.sightings)[["start", "end"]].to_numpy() if Path(args.sightings).exists() \
            else np.zeros((0, 2), int)
        self.bank = self.make_bank(args.looks)
        self.mcfg = MatchConfig()
        n = self.reader.n
        self.best = np.full(n, np.nan)
        best = self.scores.groupby("frame").score.max()
        self.best[best.index.to_numpy()] = best.to_numpy()
        self.staff = np.full(n, np.nan)
        m = self.scores.merge(self.key.track_id.rename("key_tid"), left_on="frame", right_index=True)
        m = m[m.track_id == m.key_tid]
        self.staff[m.frame.to_numpy()] = m.score.to_numpy()
        self.anchor_frames = np.flatnonzero(self.best > args.anchor_thr)
        self.hit_frames = np.flatnonzero(self.best >= args.hit_thr)
        log.info("%d boxes, %d frames; frames with a hit: %d, with an anchor: %d %s", len(self.scores), n,
                 len(self.hit_frames), len(self.anchor_frames), self.anchor_frames.tolist())
        self.cur = int(np.clip(args.start, 0, n - 1))
        self.lock: int | None = None      # followed track ID (None = best box in the frame)
        self.tab = 0                      # tab offset from the best box
        self.all_text, self.show_key = True, True
        self.click: tuple[int, int] | None = None
        self.strips = None
        self.msg = "tags found by NCC: green = anchor, orange = hit. Click a box to follow its track."

    def make_bank(self, looks: list[str]):
        """Same templates as score_tracks.py, so the CSV's template index points to them."""
        photo = pdf_jpegs(BRIEF_PDF)[0]
        drawn = draw_tag(measure_layout(to_albedo(photo, find_tag(photo))), 120)
        bank = build_bank({"drawn": drawn}, {"drawn": measure_layout(drawn)},
                          replace(BankConfig(), looks=tuple(looks)))
        self.log.info("template bank: %d templates (looks %s)", len(bank), looks)
        return bank

    def band(self, s: float) -> tuple[tuple[int, int, int], str]:
        if s > self.a.anchor_thr:
            return GREEN, "ANCHOR"
        if s >= self.a.hit_thr:
            return ORANGE, "hit"
        return (GREY, "below hit") if s >= 0 else (DARK, "negative")

    # ---------------------------------------------------------------- selection
    def rows(self) -> pd.DataFrame:
        return self.by_frame.get(self.cur, self.scores.iloc[:0])

    def selected(self) -> pd.Series | None:
        rows = self.rows()
        if not len(rows):
            return None
        if self.lock is not None:
            hit = rows[rows.track_id == self.lock]
            if len(hit):
                return hit.iloc[0]
        return rows.iloc[self.tab % len(rows)]

    def window(self, r: pd.Series) -> tuple[int, int, int, int]:
        t = self.bank[int(r.template)]
        th, tw = t.img.shape
        return int(round(r.mx - (tw - 1) / 2)), int(round(r.my - (th - 1) / 2)), tw, th

    # ---------------------------------------------------------------- drawing
    def render(self) -> np.ndarray:
        frame = self.reader.get(self.cur).copy()
        rows, sel = self.rows(), self.selected()
        if self.show_key and self.cur in self.key.index:
            g = self.key.loc[self.cur]
            cv2.rectangle(frame, (int(g.x1) - 4, int(g.y1) - 4), (int(g.x2) + 4, int(g.y2) + 4), CYAN, 1)
            cv2.putText(frame, "key: staff", (int(g.x1) - 4, int(g.y2) + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, CYAN, 1)
        for _, r in rows[::-1].iterrows():                       # best drawn last (on top)
            col, _ = self.band(r.score)
            is_sel = sel is not None and r.name == sel.name
            cv2.rectangle(frame, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), col,
                          3 if r.score > self.a.anchor_thr else (2 if r.score >= self.a.hit_thr or is_sel else 1))
            if is_sel:
                cv2.rectangle(frame, (int(r.x1) - 2, int(r.y1) - 2), (int(r.x2) + 2, int(r.y2) + 2), WHITE, 1)
            if self.all_text or r.score >= self.a.hit_thr or is_sel:
                cv2.putText(frame, f"{r.score:.3f}", (int(r.x1), max(int(r.y1) - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.42, col if r.score >= 0 else GREY, 1)
            if r.score >= self.a.hit_thr or is_sel:
                v, _ = person_view(self.reader.get(self.cur), (r.x1, r.y1, r.x2, r.y2), self.lens, self.a.margin)
                poly = to_original_points(v, rect_outline(*self.window(r)))
                cv2.polylines(frame, [poly.astype(np.int32)], True, col if r.score >= self.a.hit_thr else WHITE, 1)
        body = np.hstack([frame, self.right_panel(sel, frame.shape[0])])
        return self.with_bars(body, rows)

    def right_panel(self, r: pd.Series | None, h: int) -> np.ndarray:
        W = RIGHT_W
        panel = np.zeros((h, W, 3), np.uint8)
        if r is None:
            cv2.putText(panel, "no tracked boxes in this frame", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
            return panel
        col, name = self.band(r.score)
        v, p = person_view(self.reader.get(self.cur), (r.x1, r.y1, r.x2, r.y2), self.lens, self.a.margin)
        size = v.img.shape[0]
        s = W / size
        crop = cv2.resize(v.img, (W, W), interpolation=cv2.INTER_NEAREST)
        fx1, fy1, fx2, fy2 = self.mcfg.region
        for k in range(0, W, 8):                                   # dashed search area
            for (a, b) in (((int(fx1 * W) + k, int(fy1 * W)), (int(fx1 * W) + k + 4, int(fy1 * W))),
                           ((int(fx1 * W) + k, int(fy2 * W)), (int(fx1 * W) + k + 4, int(fy2 * W))),
                           ((int(fx1 * W), int(fy1 * W) + k), (int(fx1 * W), int(fy1 * W) + k + 4)),
                           ((int(fx2 * W), int(fy1 * W) + k), (int(fx2 * W), int(fy1 * W) + k + 4))):
                if a[0] <= int(fx2 * W) and a[1] <= int(fy2 * W):
                    cv2.line(crop, a, (min(b[0], int(fx2 * W)), min(b[1], int(fy2 * W))), YELLOW, 1)
        x0, y0, tw, th = self.window(r)
        cv2.rectangle(crop, (int(x0 * s), int(y0 * s)), (int((x0 + tw) * s) - 1, int((y0 + th) * s) - 1), col, 2)
        cv2.putText(crop, "chest search area", (int(fx1 * W) + 3, int(fy1 * W) - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, YELLOW, 1)
        panel[:W] = crop

        # what the camera saw vs the template and its two no-bars blocks, with live NCC
        gray = cv2.cvtColor(v.img, cv2.COLOR_BGR2GRAY)
        t = self.bank[int(r.template)]
        win = gray[y0:y0 + th, x0:x0 + tw].astype(np.float32)
        ok = win.shape == t.img.shape
        ncc = (lambda tt: float(cv2.matchTemplate(win, tt, cv2.TM_CCOEFF_NORMED)[0, 0])) if ok else (lambda tt: np.nan)
        raw, blocks = ncc(t.img), [ncc(tt) for tt in t.twins]
        cells = [(win if ok else np.zeros_like(t.img), True, "seen*", None), (t.img, False, "tag", raw),
                 (t.twins[0], False, "block dark", blocks[0]), (t.twins[1], False, "block white", blocks[1])]
        cw, y = (W - 5 * 6) // 4, W + 8
        for i, (img, stretch, label, val) in enumerate(cells):
            x = 6 + i * (cw + 6)
            panel[y:y + cw, x:x + cw] = fit(to_gray_bgr(img, stretch), cw, cw)
            cv2.putText(panel, label, (x, y + cw + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1)
            if val is not None:
                cv2.putText(panel, f"NCC {val:.3f}", (x, y + cw + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, YELLOW, 1)
        y += cw + 52
        live = raw - max(blocks)
        staff = self.cur in self.key.index and int(self.key.loc[self.cur].track_id) == int(r.track_id)
        lines = [
            (f"track {int(r.track_id)}  ({'followed: f frees' if self.lock is not None else 'auto: best in frame' if self.tab == 0 else 'tab'})", WHITE),
            (f"score {r.score:+.4f}  = tag {r.raw:.3f} - best block {r.twin:.3f}", col),
            (f"band: {name}   (hit >= {self.a.hit_thr}, anchor > {self.a.anchor_thr})", col),
            (f"template: {t.look} look, {t.length:.0f} px, angle {t.angle:+.0f}, {tw}x{th} px window", WHITE),
            (f"recomputed now: {live:+.4f}   crop {size}px, tilt {p['tilt_deg']:.0f} deg", GREY),
            (f"answer key: {'this IS the staff member' if staff else 'not the staff member'}", CYAN if staff else GREY),
            ("* seen = crop pixels at the match, contrast stretched", GREY),
        ]
        for text, c in lines:
            if y < h - 4:
                cv2.putText(panel, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, c, 1)
            y += 19
        return panel

    def make_strips(self, w: int) -> np.ndarray:
        """Two static strips (best of any box, staff box): bar height = score, colour = band."""
        n = self.reader.n
        out = np.full((2 * STRIP + 6, w, 3), 25, np.uint8)
        top = 70
        for row, arr in enumerate((self.best, self.staff)):
            y0 = 6 + row * STRIP
            cols = np.minimum((np.arange(n) / max(n - 1, 1) * (w - 1)).astype(int), w - 1)
            for f in np.argsort(np.nan_to_num(arr, nan=-9)):         # higher scores drawn last
                if np.isnan(arr[f]):
                    continue
                c, _ = self.band(arr[f])
                hgt = int(np.clip(arr[f] / self.a.strip_max, 0, 1) * (STRIP - 4)) + 1
                cv2.line(out, (cols[f], y0 + STRIP - 2), (cols[f], y0 + STRIP - 2 - hgt), c, 1)
            cv2.putText(out, "best box" if row == 0 else "staff box", (w - top, y0 + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, WHITE, 1)
        for s, e in self.sightings:
            x1, x2 = int(s / max(n - 1, 1) * (w - 1)), int(e / max(n - 1, 1) * (w - 1))
            cv2.rectangle(out, (x1, 0), (max(x2, x1 + 1), 3), CYAN, -1)
        return out

    def with_bars(self, body: np.ndarray, rows: pd.DataFrame) -> np.ndarray:
        w, n = body.shape[1], self.reader.n
        if self.strips is None or self.strips.shape[1] != w:
            self.strips = self.make_strips(w)
        strips = self.strips.copy()
        x = int(self.cur / max(n - 1, 1) * (w - 1))
        cv2.line(strips, (x, 0), (x, strips.shape[0]), RED, 2)
        nh = int((rows.score >= self.a.hit_thr).sum())
        na = int((rows.score > self.a.anchor_thr).sum())
        top = np.zeros((TOP, w, 3), np.uint8)
        status = (f"frame {self.cur}/{n - 1}  t={self.cur / self.reader.fps:.1f}s  |  {len(rows)} boxes, "
                  f"{nh} hits, {na} anchors  |  best {np.nan_to_num(self.best[self.cur], nan=0):+.3f}  staff "
                  f"{'-' if np.isnan(self.staff[self.cur]) else f'{self.staff[self.cur]:+.3f}'}  |  ")
        cv2.putText(top, status, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)
        x = 6 + cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0]
        for text, c in ((f"anchor > {self.a.anchor_thr}  ", GREEN), (f"hit >= {self.a.hit_thr}  ", ORANGE),
                        ("below  ", GREY), ("negative", DARK)):
            cv2.putText(top, text, (x, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)
            x += cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0]
        cv2.putText(top, self.msg[:170], (6, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.45, CYAN, 1)
        bottom = np.zeros((BOTTOM, w, 3), np.uint8)
        cv2.putText(bottom, KEY_HELP, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, GREY, 1)
        out = np.vstack([top, body, strips, bottom])
        if self.a.scale != 1.0:
            out = cv2.resize(out, None, fx=self.a.scale, fy=self.a.scale, interpolation=cv2.INTER_AREA)
        return out

    # ---------------------------------------------------------------- interaction
    def go(self, f: int) -> None:
        self.cur = int(np.clip(f, 0, self.reader.n - 1))
        self.tab = 0
        if self.lock is not None and not (self.rows().track_id == self.lock).any():
            self.msg = f"track {self.lock} is not in this frame (showing the best box)"
        else:
            self.msg = ""

    def jump(self, frames: np.ndarray, forward: bool, what: str) -> None:
        cand = frames[frames > self.cur] if forward else frames[frames < self.cur]
        if not len(cand):
            self.msg = f"no {what} {'after' if forward else 'before'} this frame"
            return
        self.go(int(cand.min() if forward else cand.max()))

    def on_mouse(self, event, x, y, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.click = (int(x / self.a.scale), int(y / self.a.scale))

    def handle_click(self) -> None:
        x, y = self.click
        self.click = None
        fh, fw = self.reader.h, self.reader.w
        if TOP <= y < TOP + fh and x < fw:
            fy = y - TOP
            rows = self.rows()
            inside = rows[(rows.x1 <= x) & (x <= rows.x2) & (rows.y1 <= fy) & (fy <= rows.y2)]
            if len(inside):
                r = inside.loc[((inside.x2 - inside.x1) * (inside.y2 - inside.y1)).idxmin()]   # smallest box
                self.lock = int(r.track_id)
                self.msg = f"following track {self.lock} (f = back to the best box)"
        elif TOP + fh <= y < TOP + fh + 2 * STRIP + 6:
            w = fw + RIGHT_W
            self.go(int(round(x / max(w - 1, 1) * (self.reader.n - 1))))

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
                self.jump(self.anchor_frames, ch == "n", "anchor")
            elif ch in (".", ","):
                self.jump(self.hit_frames, ch == ".", "hit")
            elif key == 9:
                self.lock, self.tab = None, self.tab + 1
            elif ch == "f":
                self.lock, self.tab, self.msg = None, 0, "showing the best box in each frame"
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
    ap.add_argument("--scores", default=str(TAG_SCORES), help="score_tracks.py output")
    ap.add_argument("--answer-key", default=str(ANSWER_KEY))
    ap.add_argument("--sightings", default=str(TAG_SIGHTINGS))
    ap.add_argument("--looks", nargs="+", default=["light", "dark"], help="must match score_tracks.py")
    ap.add_argument("--hit-thr", type=float, default=0.037, help="3c no_mid threshold (2 %% of other people)")
    ap.add_argument("--anchor-thr", type=float, default=0.065, help="strong-hit band (step 4 linking calibration, under review)")
    ap.add_argument("--strip-max", type=float, default=0.11, help="score at full bar height in the strips")
    ap.add_argument("--margin", type=float, default=0.15, help="crop margin (as score_tracks.py)")
    ap.add_argument("--fov-deg", type=float, default=180.0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--play-speed", type=float, default=1.0)
    ap.add_argument("--scale", type=float, default=1.0, help="window scale")
    args = ap.parse_args()

    log = setup_logging("tag_score_viewer")
    log.info("params: %s", vars(args))
    ScoreViewer(args, log).run()


if __name__ == "__main__":
    main()
