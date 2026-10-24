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
The project holds **several approaches side by side**. Each approach folder is self-contained;
only the inputs, the answer key and the environment are shared at the top level.
```
CLAUDE.md                      this file (rules for all approaches)
requirements.txt, .venv/       shared environment
AI Test Candidates.mp4, *.pdf  the task and sample video
ground_truth/                  shared answer key (staff_chain.json -> ground_truth_draft.csv),
                               so every approach is scored the same way
approach1_template_matching/   finished: YOLO + rotations, ByteTrack, tag templates, track voting
approach2_<name>/              next approach, built step by step with the user
```
Inside each approach folder:
```
README.md, JOURNEY.md, WORKLOG.md, PLAN.md   how to run / what was tried / journal / plan
detect_staff.py      command-line entry point: --video <path> --out <dir>
src/<package>/       reusable pipeline code (the final pipeline only imports from here)
scripts/             one script per step: building, checking and inspecting on the sample video
docs/                one explanation file per step (how, why, results, figures)
models/              templates / weights (weights are git-ignored)
logs/, data/         per-run logs and regenerable intermediate files (git-ignored)
outputs/             results: CSV, plots (videos git-ignored)
```
In approach 1, `src/staffid/common.py` defines `ROOT` (the approach folder), `PROJECT_ROOT`,
`SAMPLE_VIDEO` and `GT_DIR` (shared answer key). New approaches should use the same names.

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
### Approach 2
- [ ] Not started. The user wants to build it **step by step together**, understanding each step
  before moving on, so don't build ahead of the agreed step.

### Approach 1 (`approach1_template_matching/`, finished)
- [x] venv created, video and PDF analyzed, PLAN.md written
- [x] Step 1 ground truth **draft** (`ground_truth/`, needs checking by the user)
- [x] 2 person detection (YOLO11s + 4 rotations) · [x] 3 tracking (ByteTrack)
- [x] 4A tag detection (templates + surround check) · [ ] 4B trained tag classifier (optional)
- [x] 5 linking + voting · [x] 6 pixel coordinates · [x] 7 evaluation · [x] 8 CLI `detect_staff.py`
- [x] 8 robustness test written up (`docs/step8_robustness.md`); led to `min_hit_rate = 0.03`
- Main entry point: `cd approach1_template_matching && python detect_staff.py --video <file>`.
  Per-step explanations are in its `docs/`.
- Step 9 (the final 1–2 page write-up) is written **by the user**. Claude's job is to explain
  everything thoroughly in `docs/` and `WORKLOG.md` so the user has the material to write it.
