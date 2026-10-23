# Work log

## 2026-10-23: setup and Step 1 (exploration)

**Done**
- Created `.venv` (Python 3.14.7), installed `opencv-python` and `numpy`.
- Wrote `CLAUDE.md` (project rules), `PLAN.md` (plan) and `.gitignore` (excludes `data/`, `logs/`, `*.pt`, videos in `outputs/`).
- `src/staffid/common.py`: logging to the console and `logs/`, and a frame iterator.
- `src/staffid/motion.py`: MOG2 background-subtraction detector for moving people.
- `scripts/step1_explore.py`: motion boxes for every frame, and zoomed crops.
- `scripts/step1_figures.py`: exploration figures in `docs/figures/`.

**Commands**
```
.venv/bin/python scripts/step1_explore.py   # 1025/1341 frames with motion, 1720 boxes, about 4 s
.venv/bin/python scripts/step1_figures.py
```

**Key findings** (details in `docs/step1_explore.md`)
- The staff member appears in two outfits: a white shirt (about frames 330–640) and a black hoodie (about frames 690–1200). The tag is visible on both.
- The tag is about 13×7 px with bars 1–2 px wide, so it's often unreadable in a single frame. Combining evidence per person over time is needed.
- Background subtraction misses people who stand still (frames 660–810) and sometimes merges people, so YOLO is needed.
- Exposure dip at frames 1300–1305 (frame difference 99.6 against a median of 1.4).

**Problems**
- `pip install ultralytics lap` was **blocked by Claude Code's auto-mode permission check**, even though the user had approved it in chat. The user has to allow it, or run the install themselves.

**Next**
- Install `ultralytics` and `lap`, then Steps 2 and 3 (YOLO detection and tracking), then draft ground truth by hand-checking tracks.
- In the meantime: Step 4A tag detector prototype using OpenCV only.
