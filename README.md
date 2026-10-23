# Staff identification in overhead fisheye video

Finds the frames where the staff member (wearing the staff name tag) is present, and their x,y
position, in a FootfallCam overhead fisheye video.

## Setup (once)
```bash
python3 -m venv .venv            # Python 3.12+ (tested on 3.14)
.venv/bin/pip install -r requirements.txt
```
The YOLO weights (`models/yolo11s.pt`, `models/yolo11n.pt`) download automatically on the first run.
**For an offline demo, run the pipeline once beforehand** so they're cached.

## Run
```bash
source .venv/bin/activate
python detect_staff.py --video "AI Test Candidates.mp4"
python detect_staff.py --video test.mp4 --fast --no-video   # quicker: nano model, no video output
```
Results go to `outputs/<video name>/`:
| File | Content |
|---|---|
| `staff_frames.csv` | `frame, staff_present, x, y` (+ box, and source: track / interp / none) |
| `summary.json` | staff intervals (frames and seconds), counts, run time, all parameters |
| `timeline.png` | when the staff member is present |
| `annotated.mp4` | green box = staff (yellow = interpolated), grey = other people, red circle = tag hit |

Runtime on an M-series Mac: ~2.5 min for the 54 s sample video (the 25 fps video runs at ~9 fps).

## How it works
1. **Detect people:** YOLO11 on the frame rotated 0/90/180/270° (fisheye people point outwards and
   look upside down at the bottom), boxes merged with NMS.
2. **Track:** ByteTrack.
3. **Find the tag:** rotate each person upright, template-match the chest area, and keep only matches
   that pass a *surround check* (a small patch in contrast with the clothing around it).
4. **Link and vote:** join track pieces broken at the image centre (direction-consistent jumps); a
   chain with ≥ 2 tag hits is staff, and all its frames count as staff present.

Details, numbers and reasoning per step: `docs/`. Work history: `WORKLOG.md`.
