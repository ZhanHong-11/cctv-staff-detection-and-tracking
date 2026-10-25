# FootfallCam AI evaluation: staff identification

Find the frames where the staff member (wearing the staff name tag) is present in an overhead
fisheye video, and their x,y position. The task is in `AI Evaluation Test Candidates.pdf`.

## Approaches
| Folder | Idea | Status |
|---|---|---|
| [`approach1_template_matching/`](approach1_template_matching/) | YOLO on 4 rotations → ByteTrack → tag template matching with surround check → link track pieces and vote per person | finished: precision 1.0, recall 0.90, 95% of positions within 25 px (sample video, certain frames) |
| [`approach2/`](approach2/) | rectified fisheye views (4 virtual cameras) → YOLO11s → ByteTrack; user-labelled answer key via a tracker-assisted click tool; tag recognition next | in progress (see `approach2/WORKLOG.md`) |

Approach 1 has its own `README.md` (how to run), `JOURNEY.md` (what was tried and why) and `docs/`;
approach 2 so far has `WORKLOG.md` and `docs/`.

## Shared
- `AI Test Candidates.mp4`: the sample video
- `approach2/ground_truth/ground_truth.csv`: the answer key **labelled and checked by the user**
  (staff present in all 1341 frames). Use this one for scoring every approach.
- `approach1_groundtruth/`: approach 1's earlier answer key, a draft made by Claude, never checked.
- `models/`: shared YOLO weights (git-ignored, download on first run) and approach 1's tag templates
- `requirements.txt` + `.venv/`: one Python environment for all approaches

## Setup (once)
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```
