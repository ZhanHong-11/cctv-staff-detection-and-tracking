# FootfallCam AI evaluation: staff identification

Find the frames where the staff member (wearing the staff name tag) is present in an overhead
fisheye video, and their x,y position. The task is in `AI Evaluation Test Candidates.pdf`.

## Approaches
| Folder | Idea | Status |
|---|---|---|
| [`approach1_template_matching/`](approach1_template_matching/) | YOLO on 4 rotations → ByteTrack → tag template matching with surround check → link track pieces and vote per person | finished: precision 1.0, recall 0.90, 95% of positions within 25 px (sample video, certain frames) |
| `approach2_…` | to be decided (options A–D listed in `CLAUDE.md`) | not started |

Each folder has its own `README.md` (how to run), `JOURNEY.md` (what was tried and why) and `docs/`.

## Shared
- `AI Test Candidates.mp4`: the sample video
- `ground_truth/`: the answer key, a **draft that needs checking by the user**. Every approach is
  scored against the same key.
- `requirements.txt` + `.venv/`: one Python environment for all approaches

## Setup (once)
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```
