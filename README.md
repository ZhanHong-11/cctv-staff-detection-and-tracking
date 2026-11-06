# FootfallCam AI evaluation: staff identification

Find the frames where the staff member (wearing the staff name tag) is present in an overhead
fisheye video, and their x,y position. The task is in `AI Evaluation Test Candidates.pdf`.

## Approaches
| Folder | Idea | Status |
|---|---|---|
| [`approach2/`](approach2/) | rectified fisheye views (4 virtual cameras) → YOLO11s → ByteTrack → NCC tag score (templates from the brief's tag photo) → live staff following (confirm, follow, movement guess, suspects) | **the solution**: use this one |
| [`approach1/`](approach1/) | YOLO on 4 rotations → ByteTrack → tag template matching with surround check → link track pieces and vote per person | **first experiment, kept for reference only; not maintained, not for use** |

Approach 2 is documented in `approach2/WORKLOG.md` (journal) and `approach2/docs/` (one file per
step). Approach 1 keeps its own `README.md`, `JOURNEY.md` and `docs/` as a record of what was tried
first and why approach 2 replaced it.

## Run the solution
```bash
cd approach2
../.venv/bin/python staff_detector.py --video <file.mp4>              # -> result/<video name>/staff.csv, staff.mp4, ...
../.venv/bin/python staff_detector.py --video <file.mp4> --no-video   # CSVs only, faster
```
The answer is `result/<video name>/staff.csv` (`frame,staff_present,x,y`); see
`approach2/docs/step6_staff_detector.md`.

## Shared
- `AI Test Candidates.mp4`: the sample video
- `approach2/ground_truth/ground_truth.csv`: the answer key **labelled and checked by the user**
  (staff present in all 1341 frames). Use this one for scoring every approach.
- `approach1/groundtruth/`: approach 1's earlier answer key, a draft made by Claude, never checked
  (reference only, do not use for scoring).
- `models/`: shared YOLO weights (git-ignored, download on first run) and approach 1's tag templates
- `requirements.txt` + `.venv/`: one Python environment for all approaches

## Setup (once)
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```
