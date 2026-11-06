"""Step 3a, part 2: label where the staff member's tag is visible (user tool).

Shows the staff member (from the answer key) as a person-centred rectified crop, enlarged with
a pixel grid. You mark SIGHTINGS (frame ranges where the tag is visible) and drag a box around
the tag on a few frames of each sighting (e.g. start, middle, end). Skip passes that look the
same as one you already labelled: frames outside every sighting mean "not labelled", NOT
"no tag".

These labels are for measuring the real tag (size, angle, contrast) and for testing the tag
detectors; they are never used as training data or templates.

Keys
  drag (left panel)  tag box on this frame (snapped to whole pixels; replaces this frame's box)
  i / k  j / l       move the box up / down, left / right by 1 px
  I / K  J / L       box 1 px shorter / taller, narrower / wider
  x                  delete this frame's box
  s / e              sighting starts / ends here (e closes the range from the last s)
  r                  remove the sighting that contains this frame
  [ / ]              previous / next sighting          , / .   previous / next box
  a / d              previous / next frame             z / c   back / forward 1 second
  + / -              zoom in / out (centred on the box, or the chest if there is none yet)
  p                  play (any key stops)              t       pixel grid on / off
  u                  undo                              w       save       q / Esc  save and quit

Right side: the whole frame (staff box coloured by answer-key source, tag box in red) and the
original (not rectified) crop with the tag box mapped onto it. Bottom strip: timeline with
sightings (green), boxes (white) and the current frame (red).

Saves ground_truth/tag_labels.json (all labels, reloaded next time), tag_boxes.csv (one row per
box: crop box, size, camera aim, position in the original frame) and tag_sightings.csv.

Usage (from approach2/):
    ../.venv/bin/python scripts/label_tag.py
    ../.venv/bin/python scripts/label_tag.py --start 370 --view 640 --scale 0.9
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from staffid2.common import ROOT, SAMPLE_VIDEO, setup_logging  # noqa: E402
from staffid2.drawing import CYAN, GREEN, GREY, RED, SOURCE_COLOR, WHITE, YELLOW, draw_box  # noqa: E402
from staffid2.person_crop import Lens, aim, original_crop, person_view, to_original_points  # noqa: E402

WIN = "label tag"
TOP, TIMELINE, BOTTOM = 44, 16, 36
RIGHT_W, RIGHT_TOP_H = 480, 360
KEY_HELP = ("drag=tag box  ijkl=move  IJKL=size  x=del box  s/e=sighting start/end  r=remove sighting  "
            "[ ]=sighting  , .=box  a/d z/c=move  +/-=zoom  p=play  t=grid  u=undo  w=save  q=quit")
EDGE_PTS = 5  # points per box edge when mapping the tag box to the original frame


class FrameReader:
    """Random access to video frames; sequential reads avoid a seek (copied from label_staff.py)."""

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


def fit(img: np.ndarray, w: int, h: int) -> tuple[np.ndarray, float]:
    """img scaled to fit w x h (aspect kept, nearest so pixels stay visible), padded black."""
    s = min(w / img.shape[1], h / img.shape[0])
    small = cv2.resize(img, (max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s))),
                       interpolation=cv2.INTER_NEAREST if s >= 1 else cv2.INTER_AREA)
    out = np.zeros((h, w, 3), np.uint8)
    out[:small.shape[0], :small.shape[1]] = small
    return out, s


def box_outline(b: dict) -> np.ndarray:
    """Points around a crop box (pixel-edge coords) in remap index coords (pixel centre = integer)."""
    t = np.linspace(0, 1, EDGE_PTS)
    x1, y1, x2, y2 = b["x1"] - 0.5, b["y1"] - 0.5, b["x2"] - 0.5, b["y2"] - 0.5
    xs = np.concatenate([x1 + (x2 - x1) * t, np.full_like(t, x2), x2 - (x2 - x1) * t, np.full_like(t, x1)])
    ys = np.concatenate([np.full_like(t, y1), y1 + (y2 - y1) * t, np.full_like(t, y2), y2 - (y2 - y1) * t])
    return np.stack([xs, ys], axis=1)


class TagLabeller:
    def __init__(self, args: argparse.Namespace, log):
        self.a, self.log = args, log
        self.reader = FrameReader(args.video)
        self.lens = Lens.for_frame(self.reader.w, self.reader.h, args.fov_deg)
        key = pd.read_csv(args.answer_key)
        self.key = key[(key.staff_present == 1) & key.x1.notna()].set_index("frame")
        self.frames = self.key.index.to_numpy()
        self.out_dir = Path(args.out_dir)
        self.state_path = self.out_dir / "tag_labels.json"
        self.sightings: list[list[int]] = []
        self.boxes: dict[int, dict] = {}
        if self.state_path.exists():
            st = json.loads(self.state_path.read_text())
            self.sightings = [list(s) for s in st["sightings"]]
            self.boxes = {int(b["frame"]): {k: b[k] for k in ("x1", "y1", "x2", "y2")} for b in st["boxes"]}
            log.info("resumed %d sightings and %d boxes from %s", len(self.sightings), len(self.boxes), self.state_path)
            self.check_resumed(st["boxes"])
        self.history: list[tuple] = []
        self.pending: int | None = None
        self.cur = int(self.frames[np.searchsorted(self.frames, min(args.start, self.frames[-1]))])
        self.grid = True
        self.zoom = args.zoom
        self.drag: list | None = None   # [x0, y0, x1, y1] in display px of the left panel
        self.drag_done = False
        self.cache_frame, self.cache_view = -1, None
        self.msg = "find a frame where the tag shows: s = sighting start, drag a box around the tag"

    def check_resumed(self, boxes: list[dict]) -> None:
        """Crop boxes are only valid for the same staff box: warn if the answer key changed."""
        for b in boxes:
            f = int(b["frame"])
            if f not in self.key.index:
                self.log.warning("box at frame %d: staff no longer in the answer key", f)
                continue
            g = self.key.loc[f]
            if max(abs(g.x1 - b["staff_x1"]), abs(g.y1 - b["staff_y1"]), abs(g.x2 - b["staff_x2"]),
                   abs(g.y2 - b["staff_y2"])) > 0.5:
                self.log.warning("box at frame %d: answer-key box changed, crop box may be off", f)

    # ---------------------------------------------------------------- state
    def push(self) -> None:
        self.history.append((copy.deepcopy(self.sightings), copy.deepcopy(self.boxes), self.pending))

    def undo(self) -> None:
        if not self.history:
            self.msg = "nothing to undo"
            return
        self.sightings, self.boxes, self.pending = self.history.pop()
        self.msg = "undone"
        self.save(quiet=True)

    def sighting_at(self, f: int) -> int | None:
        for k, (s, e) in enumerate(self.sightings):
            if s <= f <= e:
                return k
        return None

    def view(self, f: int):
        if f != self.cache_frame:
            g = self.key.loc[f]
            self.cache_view = person_view(self.reader.get(f), (g.x1, g.y1, g.x2, g.y2), self.lens, self.a.margin)
            self.cache_frame = f
        return self.cache_view

    def window(self, size: int) -> tuple[int, int, int]:
        """Part of the crop shown in the left panel (x, y, side) at the current zoom. Centred on
        this frame's box, else the nearest box (same relative place), else the chest area."""
        side = max(4, int(round(size / self.zoom)))
        if side >= size:
            return 0, 0, size
        if self.cur in self.boxes:
            b, rel = self.boxes[self.cur], 1.0
        elif self.boxes:
            near = min(self.boxes, key=lambda f: abs(f - self.cur))
            b, rel = self.boxes[near], size / self.box_crop_size(near)
        else:
            b, rel = None, 1.0
        cx, cy = ((b["x1"] + b["x2"]) / 2 * rel, (b["y1"] + b["y2"]) / 2 * rel) if b else \
            (size * self.a.chest_x, size * self.a.chest_y)
        wx = int(np.clip(round(cx - side / 2), 0, size - side))
        wy = int(np.clip(round(cy - side / 2), 0, size - side))
        return wx, wy, side

    def box_crop_size(self, f: int) -> int:
        g = self.key.loc[f]
        return aim((g.x1, g.y1, g.x2, g.y2), self.lens, self.a.margin)["size"]

    def move(self, step: int) -> None:
        i = int(np.searchsorted(self.frames, self.cur)) + step
        self.cur = int(self.frames[int(np.clip(i, 0, len(self.frames) - 1))])

    # ---------------------------------------------------------------- actions
    def set_box(self, b: dict) -> None:
        size = self.view(self.cur)[1]["size"]
        b = {k: int(np.clip(v, 0, size)) for k, v in b.items()}
        if b["x2"] - b["x1"] < 1 or b["y2"] - b["y1"] < 1:
            self.msg = "box too small"
            return
        self.push()
        self.boxes[self.cur] = b
        where = "" if self.sighting_at(self.cur) is not None or self.pending is not None else \
            "  (outside any sighting: press s/e to mark the range)"
        self.msg = f"box {b['x2'] - b['x1']}x{b['y2'] - b['y1']} px at frame {self.cur}{where}"
        self.save(quiet=True)

    def nudge(self, dx1: int, dy1: int, dx2: int, dy2: int) -> None:
        b = self.boxes.get(self.cur)
        if b is None:
            self.msg = "no box on this frame"
            return
        self.set_box(dict(x1=b["x1"] + dx1, y1=b["y1"] + dy1, x2=b["x2"] + dx2, y2=b["y2"] + dy2))

    def end_sighting(self) -> None:
        if self.pending is None:
            self.msg = "press s at the sighting's first frame first"
            return
        s, e = sorted((self.pending, self.cur))
        clash = [k for k, (a, b) in enumerate(self.sightings) if a <= e and s <= b]
        if clash:
            self.msg = f"{s}-{e} overlaps sighting {self.sightings[clash[0]]}: r removes it first"
            return
        self.push()
        self.sightings = sorted(self.sightings + [[s, e]])
        self.pending = None
        n_box = sum(s <= f <= e for f in self.boxes)
        self.msg = f"sighting {s}-{e} ({e - s + 1} frames, {n_box} boxes)" + (
            "  drag boxes on a few frames of it" if n_box < 3 else "")
        self.save(quiet=True)

    def jump(self, targets: list[int], forward: bool, what: str) -> None:
        cand = [t for t in targets if (t > self.cur if forward else t < self.cur)]
        if not cand:
            self.msg = f"no {what} {'after' if forward else 'before'} this frame"
            return
        self.cur = min(cand) if forward else max(cand)
        self.msg = ""

    # ---------------------------------------------------------------- drawing
    def render(self) -> np.ndarray:
        f, d = self.cur, self.a.view
        v, p = self.view(f)
        wx, wy, side = self.window(p["size"])
        sd = d / side
        left = cv2.resize(v.img[wy:wy + side, wx:wx + side], (d, d), interpolation=cv2.INTER_NEAREST)
        if self.grid and sd >= self.a.grid_min_px:
            lines = left.copy()
            for k in range(1, side):
                q = int(round(k * sd))
                cv2.line(lines, (q, 0), (q, d), (40, 40, 40), 1)
                cv2.line(lines, (0, q), (d, q), (40, 40, 40), 1)
            left = cv2.addWeighted(lines, 0.35, left, 0.65, 0)
        if self.zoom > 1:
            cv2.putText(left, f"zoom x{self.zoom}", (6, d - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW, 1)
        b = self.boxes.get(f)
        if b:
            cv2.rectangle(left, (int((b["x1"] - wx) * sd), int((b["y1"] - wy) * sd)),
                          (int((b["x2"] - wx) * sd) - 1, int((b["y2"] - wy) * sd) - 1), RED, 2)
        if self.drag:
            x0, y0, x1, y1 = self.drag
            cv2.rectangle(left, (x0, y0), (x1, y1), CYAN, 1)

        frame = self.reader.get(f).copy()
        g = self.key.loc[f]
        color = SOURCE_COLOR.get(g.source, WHITE)
        draw_box(frame, (g.x1, g.y1, g.x2, g.y2), color, thickness=2)
        orig, (ox, oy) = original_crop(self.reader.get(f), (g.x1, g.y1, g.x2, g.y2), self.a.margin)
        orig = orig.copy()
        if b:
            poly = to_original_points(v, box_outline(b))
            cv2.polylines(frame, [poly.astype(np.int32)], True, RED, 2)
            cv2.polylines(orig, [(poly - (ox, oy)).astype(np.int32)], True, RED, 1)
        top_r, _ = fit(frame, RIGHT_W, RIGHT_TOP_H)
        bot_r, _ = fit(orig, RIGHT_W, d - RIGHT_TOP_H)
        cv2.putText(bot_r, "original crop", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, YELLOW, 1)
        body = np.hstack([left, np.vstack([top_r, bot_r])])
        return self.with_bars(body, g, p)

    def timeline(self, w: int) -> np.ndarray:
        bar = np.full((TIMELINE, w, 3), 30, np.uint8)
        n = self.reader.n
        px = lambda i: int(i / max(n - 1, 1) * (w - 1))  # noqa: E731
        for s, e in self.sightings:
            cv2.rectangle(bar, (px(s), 2), (max(px(e), px(s) + 1), TIMELINE - 3), GREEN, -1)
        if self.pending is not None:
            cv2.rectangle(bar, (px(min(self.pending, self.cur)), 5),
                          (max(px(max(self.pending, self.cur)), px(min(self.pending, self.cur)) + 1), TIMELINE - 6),
                          YELLOW, -1)
        for f in self.boxes:
            cv2.line(bar, (px(f), 0), (px(f), 4), WHITE, 1)
        cv2.line(bar, (px(self.cur), 0), (px(self.cur), TIMELINE), RED, 2)
        return bar

    def with_bars(self, body: np.ndarray, g: pd.Series, p: dict) -> np.ndarray:
        f, w = self.cur, body.shape[1]
        k = self.sighting_at(f)
        sight = f"sighting {self.sightings[k][0]}-{self.sightings[k][1]}" if k is not None else "no sighting"
        if self.pending is not None:
            sight += f"  (open from {self.pending})"
        status = (f"frame {f}/{self.reader.n - 1}  t={f / self.reader.fps:.1f}s  staff box: {g.source}"
                  f"{'  (straight-line fill, crop may be off)' if g.source == 'interp' else ''}  |  {sight}  |  "
                  f"crop {p['size']}px tilt {p['tilt_deg']:.0f}  |  {len(self.sightings)} sightings, "
                  f"{len(self.boxes)} boxes")
        top = np.zeros((TOP, w, 3), np.uint8)
        cv2.putText(top, status[:170], (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)
        cv2.putText(top, self.msg[:170], (6, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.45, CYAN, 1)
        bottom = np.zeros((BOTTOM, w, 3), np.uint8)
        half = len(KEY_HELP) // 2
        cut = KEY_HELP.rfind("  ", 0, half + 20)
        cv2.putText(bottom, KEY_HELP[:cut], (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, GREY, 1)
        cv2.putText(bottom, KEY_HELP[cut + 2:], (6, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, GREY, 1)
        out = np.vstack([top, body, self.timeline(w), bottom])
        if self.a.scale != 1.0:
            out = cv2.resize(out, None, fx=self.a.scale, fy=self.a.scale, interpolation=cv2.INTER_AREA)
        return out

    def play(self) -> None:
        delay = max(1, int(1000 / (self.reader.fps * self.a.play_speed)))
        while self.cur < self.frames[-1]:
            self.move(1)
            cv2.imshow(WIN, self.render())
            if cv2.waitKey(delay) & 0xFF != 255:
                break
        self.msg = f"stopped at {self.cur}"

    # ---------------------------------------------------------------- saving
    def box_rows(self) -> pd.DataFrame:
        rows = []
        for f in sorted(self.boxes):
            b = self.boxes[f]
            g = self.key.loc[f]
            v, p = person_view(self.reader.get(f), (g.x1, g.y1, g.x2, g.y2), self.lens, self.a.margin)
            poly = to_original_points(v, box_outline(b))
            centre = to_original_points(v, [[(b["x1"] + b["x2"]) / 2 - 0.5, (b["y1"] + b["y2"]) / 2 - 0.5]])[0]
            k = self.sighting_at(f)
            rows.append(dict(frame=f, sighting=-1 if k is None else k, **b, w=b["x2"] - b["x1"], h=b["y2"] - b["y1"],
                             crop_size=p["size"], azimuth_deg=p["azimuth_deg"], tilt_deg=p["tilt_deg"],
                             fov_deg=p["fov_deg"], margin=self.a.margin,
                             orig_x=centre[0], orig_y=centre[1], orig_x1=poly[:, 0].min(), orig_y1=poly[:, 1].min(),
                             orig_x2=poly[:, 0].max(), orig_y2=poly[:, 1].max(), staff_source=g.source,
                             staff_x1=g.x1, staff_y1=g.y1, staff_x2=g.x2, staff_y2=g.y2))
        return pd.DataFrame(rows)

    def save(self, quiet: bool = False) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        boxes = self.box_rows()
        meta = dict(video=self.a.video, answer_key=self.a.answer_key, margin=self.a.margin, fov_deg=self.a.fov_deg,
                    saved=time.strftime("%Y-%m-%d %H:%M:%S"), labelled_by="user (label_tag.py)",
                    note="frames outside every sighting are NOT labelled (not 'no tag')",
                    sightings=self.sightings, boxes=boxes.round(2).to_dict("records"))
        self.state_path.write_text(json.dumps(meta, indent=1))
        boxes.round(2).to_csv(self.out_dir / "tag_boxes.csv", index=False)
        sight = pd.DataFrame([dict(sighting=k, start=s, end=e, n_frames=e - s + 1,
                                   n_boxes=int(((boxes.frame >= s) & (boxes.frame <= e)).sum()) if len(boxes) else 0)
                              for k, (s, e) in enumerate(self.sightings)],
                             columns=["sighting", "start", "end", "n_frames", "n_boxes"])
        sight.to_csv(self.out_dir / "tag_sightings.csv", index=False)
        if not quiet:
            self.log.info("saved %s, tag_boxes.csv (%d), tag_sightings.csv (%d)", self.state_path, len(boxes), len(sight))
            self.msg = "saved"

    def summary(self) -> None:
        boxes = self.box_rows()
        self.log.info("sightings (%d): %s", len(self.sightings), self.sightings)
        self.log.info("frames inside sightings: %d", sum(e - s + 1 for s, e in self.sightings))
        if len(boxes):
            self.log.info("boxes: %d; per sighting: %s; outside any sighting: %s", len(boxes),
                          boxes[boxes.sighting >= 0].groupby("sighting").size().to_dict(),
                          boxes[boxes.sighting < 0].frame.tolist())
            self.log.info("tag box w px: min %d median %.0f max %d; h px: min %d median %.0f max %d",
                          boxes.w.min(), boxes.w.median(), boxes.w.max(), boxes.h.min(), boxes.h.median(), boxes.h.max())
        empty = [s for s in self.sightings if not any(s[0] <= f <= s[1] for f in self.boxes)]
        if empty:
            self.log.warning("sightings without any box: %s", empty)

    # ---------------------------------------------------------------- main loop
    def on_mouse(self, event, x, y, flags, param) -> None:
        x, y = int(x / self.a.scale), int(y / self.a.scale) - TOP
        d = self.a.view
        x, y = int(np.clip(x, 0, d - 1)), int(np.clip(y, 0, d - 1))
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag = [x, y, x, y]
        elif event == cv2.EVENT_MOUSEMOVE and self.drag and flags & cv2.EVENT_FLAG_LBUTTON:
            self.drag[2:] = [x, y]
        elif event == cv2.EVENT_LBUTTONUP and self.drag:
            self.drag[2:] = [x, y]
            self.drag_done = True

    def finish_drag(self) -> None:
        x0, y0, x1, y1 = self.drag
        self.drag, self.drag_done = None, False
        if abs(x1 - x0) < 3 and abs(y1 - y0) < 3:
            return
        wx, wy, side = self.window(self.view(self.cur)[1]["size"])
        s = side / self.a.view   # display px -> crop px
        self.set_box(dict(x1=wx + int(np.floor(min(x0, x1) * s)), y1=wy + int(np.floor(min(y0, y1) * s)),
                          x2=wx + int(np.ceil(max(x0, x1) * s)), y2=wy + int(np.ceil(max(y0, y1) * s))))

    def run(self) -> None:
        cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WIN, self.on_mouse)
        fps = int(round(self.reader.fps))
        while True:
            cv2.imshow(WIN, self.render())
            key = cv2.waitKey(30) & 0xFF
            if self.drag_done:
                self.finish_drag()
                continue
            if key == 255:
                continue
            ch = chr(key) if key < 128 else ""
            nudges = {"i": (0, -1, 0, -1), "k": (0, 1, 0, 1), "j": (-1, 0, -1, 0), "l": (1, 0, 1, 0),
                      "I": (0, 0, 0, -1), "K": (0, 0, 0, 1), "J": (0, 0, -1, 0), "L": (0, 0, 1, 0)}
            if ch in nudges:
                self.nudge(*nudges[ch])
            elif ch == "x":
                if self.cur in self.boxes:
                    self.push()
                    del self.boxes[self.cur]
                    self.msg = f"box at {self.cur} deleted"
                    self.save(quiet=True)
            elif ch == "s":
                self.push()
                self.pending = self.cur
                self.msg = f"sighting open from {self.cur}: move to its last frame and press e"
            elif ch == "e":
                self.end_sighting()
            elif ch == "r":
                k = self.sighting_at(self.cur)
                if k is None:
                    self.msg = "no sighting here"
                else:
                    self.push()
                    self.msg = f"sighting {self.sightings.pop(k)} removed (its boxes stay; x deletes a box)"
                    self.save(quiet=True)
            elif ch in "[]" and ch:
                self.jump([s for s, _ in self.sightings], ch == "]", "sighting")
            elif ch in ",." and ch:
                self.jump(list(self.boxes), ch == ".", "box")
            elif ch in "adzc" and ch:
                self.move({"a": -1, "d": 1, "z": -fps, "c": fps}[ch])
                self.msg = ""
            elif ch == "p":
                self.play()
            elif ch == "t":
                self.grid = not self.grid
            elif ch in "+=-" and ch:
                self.zoom = int(np.clip(self.zoom + (-1 if ch == "-" else 1), 1, self.a.max_zoom))
                self.msg = f"zoom x{self.zoom}"
            elif ch == "u":
                self.undo()
            elif ch == "w":
                self.save()
            elif ch == "q" or key == 27:
                break
        cv2.destroyAllWindows()
        self.save()
        self.summary()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", default=str(SAMPLE_VIDEO))
    ap.add_argument("--answer-key", default=str(ROOT / "ground_truth" / "ground_truth.csv"))
    ap.add_argument("--start", type=int, default=0, help="first frame to show")
    ap.add_argument("--margin", type=float, default=0.15, help="extra view around the person box (as staff_crops.py)")
    ap.add_argument("--fov-deg", type=float, default=180.0, help="lens field of view across 2*radius")
    ap.add_argument("--view", type=int, default=720, help="size of the enlarged crop panel (px)")
    ap.add_argument("--grid-min-px", type=float, default=4.0, help="draw the pixel grid when 1 px is at least this big")
    ap.add_argument("--zoom", type=int, default=2, help="starting zoom of the left panel (+/- to change)")
    ap.add_argument("--max-zoom", type=int, default=4)
    ap.add_argument("--chest-x", type=float, default=0.5, help="zoom centre when no box exists yet (fraction of crop)")
    ap.add_argument("--chest-y", type=float, default=0.4)
    ap.add_argument("--play-speed", type=float, default=1.0, help="playback speed (1 = real time)")
    ap.add_argument("--scale", type=float, default=1.0, help="window scale")
    ap.add_argument("--out-dir", default=str(ROOT / "ground_truth"))
    args = ap.parse_args()
    if args.view <= RIGHT_TOP_H:
        ap.error(f"--view must be larger than {RIGHT_TOP_H}")

    log = setup_logging("label_tag")
    log.info("params: %s", vars(args))
    TagLabeller(args, log).run()


if __name__ == "__main__":
    main()
