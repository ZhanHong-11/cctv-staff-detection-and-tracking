# FootfallCam AI Evaluation: Staff Identification

Interview take-home task. Find the frames where the **staff member** (the person wearing a white
name tag with black vertical bars) appears in an overhead fisheye video, and give their x,y position.
The full brief is in `AI Evaluation Test Candidates.pdf` and the step-by-step plan is in `PLAN.md`.

## Input data
- `AI Test Candidates.mp4`: 960×720, 25 fps, 1341 frames (about 54 s). Overhead fisheye view of an office.
  Most people sit still at desks and a few walk the central corridor.
- The tag is tiny (about 8–20 px) and often hidden, so a person is labeled staff for their whole
  track, not only in the frames where the tag is visible.
- The code must also work on an **unseen test video** in the interview. Never hard-code frame
  numbers, regions or colors taken from this video.

## Environment rules (strict)
- Use only the project venv: `.venv/` (Python 3.14). Run tools as `.venv/bin/python` or
  `source .venv/bin/activate && ...`. Never use the system Python or install anything globally.
- **Ask the user before installing any package** (`pip install`) and before any network
  download (e.g. model weights). Once a package is approved and installed, add it to `requirements.txt` with its version pinned.
- Already installed: `opencv-python`, `numpy`.
- `ffmpeg` is not available. Read and write video with OpenCV (`cv2.VideoWriter`, `mp4v` codec).
- Put temporary or throwaway files in the session scratchpad, not in the project.

## Directory layout
```
CLAUDE.md            this file
PLAN.md              step-by-step plan (what and why)
WORKLOG.md           running journal: what was done, the commands run, results and decisions
requirements.txt     pinned dependencies
src/staffid/         reusable pipeline code (detection, tracking, tag detection, voting, I/O)
scripts/             one script per plan step, e.g. step1_extract_frames.py
detect_staff.py      final command-line entry point: --video <path> --out <dir>
docs/                one explanation file per step: stepN_<name>.md (how, why, results, figures)
logs/                log file for each run: stepN_<name>_<YYYYmmdd-HHMMSS>.log
data/                generated data (frames, crops, labels), can be regenerated
outputs/             results: CSV, annotated video, plots
```

## Logging and documentation conventions
- Every script logs with Python's `logging` module to both the console and `logs/...log`. Log
  the parameters, the input, the output paths, the counts and the timings.
- After each step, add an entry to `WORKLOG.md` (date, step, what was done, the commands run,
  key numbers, any problems, next step) and write or update `docs/stepN_<name>.md`.
- Docs explain **how** and **why**, include figures saved in `docs/figures/`, and state the
  assumptions and limitations honestly. If something wasn't verified, say so.
- Ground truth that Claude produced must be marked as such, and the user must check it before
  it's used in final metrics.

## Code conventions
- Python 3, type hints, small functions, no notebooks for core logic.
- Parameters go in argparse flags or a config dict, never as magic numbers buried in the code.
- Output CSV format: `frame,staff_present,x,y` with pixel coordinates and the origin at the top-left.
- Keep it fast enough on a laptop CPU for the live demo.

## Status
- [x] venv created, video and PDF analyzed, PLAN.md written
- [x] Step 1 ground truth **draft** (`ground_truth/`, needs checking by the user)
- [x] 2 person detection (YOLO11s + 4 rotations) · [x] 3 tracking (ByteTrack)
- [x] 4A tag detection (templates + surround check) · [ ] 4B trained tag classifier (optional)
- [x] 5 linking + voting · [x] 6 pixel coordinates · [x] 7 evaluation · [x] 8 CLI `detect_staff.py`
- [x] 8 robustness test written up (`docs/step8_robustness.md`); led to `min_hit_rate = 0.03`
- Main entry point: `python detect_staff.py --video <file>`. Per-step explanations are in `docs/`.
- Step 9 (the final 1–2 page write-up) is written **by the user**. Claude's job is to explain
  everything thoroughly in `docs/` and `WORKLOG.md` so the user has the material to write it.
