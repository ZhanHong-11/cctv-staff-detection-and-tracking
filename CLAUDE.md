# FootfallCam AI Evaluation: Staff Identification

Interview take-home task. Find the frames where the **staff member** (the person wearing a white
name tag with black vertical bars) appears in an overhead fisheye video, and give their x,y position.
The full brief is in `AI Evaluation Test Candidates.pdf`. Each approach has its own plan, e.g.
`approach1_template_matching/PLAN.md`.

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
- Already installed (pinned in `requirements.txt`): opencv-python, numpy, ultralytics, lap, torch,
  torchvision, pandas, matplotlib, tqdm. `torch.backends.mps` (Mac GPU) works. Not installed:
  `sahi` (on standby).
- Claude Code's auto-mode check blocked `pip install` once, even with the user's approval. If
  that happens, add the package to `requirements.txt` and ask the user to run
  `.venv/bin/pip install -r requirements.txt` themselves.
- `ffmpeg` is not available. Read and write video with OpenCV. Write videos as H.264
  (`avc1`, plays in QuickTime) and fall back to `mp4v` if `avc1` fails to open.
- Ultralytics usage analytics are turned off in code (`SETTINGS.update({"sync": False})`).
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

## Key facts about the sample video (learned in approach 1, useful for any approach)
- The staff member **changes clothes**: white shirt + black trousers, then puts on a black hoodie
  at frames ~633–653 while standing still. Clothing can't identify staff; only the tag can.
- The tag is ~13×7 px (bars 1–2 px). It's visible in only ~18% of the staff member's frames.
- People point **away from the image centre** (upside down at the bottom). Plain YOLO misses most
  of them; running it on 4 rotations fixes this.
- **The image centre is a blind spot**: people are seen from straight above and aren't detected,
  so tracks break there every time someone crosses.
- A **second dark-clothed person (not staff)** walks near the staff member around frames
  900–1210; the tracker's ID swaps onto them at ~1211.
- **Exposure dip** at frames 1300–1305 (the image goes nearly black).
- The staff member is present in almost every frame, so presence metrics look good even with
  the wrong person. **Position accuracy is what reveals a wrong identity.**
- The answer key's positions are the tracker's own boxes, so a median error of 0 px means "same box",
  not "tight fit".

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
- **Waiting on two decisions from the user** (offered on 2026-10-24):
  1. Which idea:
     - **A. Trained tag classifier** (recommended): real tag patches + synthetic tags pasted
       onto other people → small CNN replaces the hand-set surround-check rules.
     - **B. YOLO tag detector on whole frames**: tag first, then attach it to the nearest person.
     - **C. Unwarp the fisheye first** so ordinary detectors work.
     - **D. Appearance re-identification**: confirm staff once by the tag, then follow by
       appearance (weak here because the staff member changes clothes).
  2. Reuse approach 1's detection + tracking (faster, clean comparison) **or** build everything
     from scratch (more learning).
- Once decided: create `approach2_<name>/` with the same layout, write the plan for **step 1
  only**, and go through it with the user before building.

### Approach 1 (`approach1_template_matching/`, finished)
- [x] venv created, video and PDF analyzed, PLAN.md written
- [x] Step 1 ground truth **draft** (`ground_truth/`, needs checking by the user)
- [x] 2 person detection (YOLO11s + 4 rotations) · [x] 3 tracking (ByteTrack)
- [x] 4A tag detection (templates + surround check) · [ ] 4B trained tag classifier (optional)
- [x] 5 linking + voting · [x] 6 pixel coordinates · [x] 7 evaluation · [x] 8 CLI `detect_staff.py`
- [x] 8 robustness test written up (`docs/step8_robustness.md`); led to `min_hit_rate = 0.03`
- Main entry point: `cd approach1_template_matching && python detect_staff.py --video <file>`.
  Per-step explanations are in its `docs/`, and the story of what was tried is in its `JOURNEY.md`.
- Result (sample video, certain frames): precision 1.0, recall 0.90, 95% of positions within 25 px,
  staff present frames 365–1246, ~2.5 min runtime.
- Known gaps: white-shirt walk-off (275–364) missed; ID swap puts the box on the wrong person at
  ~1211–1246; seated period 0–274 undecidable; never tested on a video without staff; all
  settings tuned on one clip.
- Still open for the user: **check the answer key** (`ground_truth/staff_chain.json`: is the
  seated person in frames 0–274 the staff member? do they leave at ~1257?).
- Verified after the move into `approach1_template_matching/` (2026-10-24): `scripts/step5_vote.py`
  gives the same result as before.
- Step 9 (the final 1–2 page write-up) is written **by the user**. Claude's job is to explain
  everything thoroughly in `docs/` and `WORKLOG.md` so the user has the material to write it.
