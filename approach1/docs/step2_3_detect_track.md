# Steps 2–3: Detecting and tracking people

**Code:** `src/staffid/detect.py`, `src/staffid/track.py`, `scripts/step2_3_detect_track.py`
**Run:** `python scripts/step2_3_detect_track.py` (~90 s for the sample video on an M-series Mac)
**Output:** `data/step2_3/tracks.csv` (frame, track_id, box, confidence), `track_summary.csv`, `tracks.mp4`

## 1. Person detection: YOLO11 + rotations

### The problem: COCO detectors expect upright people
YOLO11 is trained on COCO, which is mostly side-view photos of upright people. In an overhead
fisheye view a standing person points **away from the image centre**:
- at the **top** of the image they look upright → detected
- at the **sides** they lie sideways → often missed
- at the **bottom** they're upside down → almost always missed

A first test confirmed this: plain YOLO found only **0–5 of ~20 people** per frame, all in the top part.
At frames 750 and 950 it missed the staff member completely (bottom half).

### The fix: test-time rotation
Each frame is rotated by 0°, 90°, 180° and 270°, and YOLO runs on all four. Every person is upright
in at least one rotation. The boxes are rotated back to the original frame (`unrotate_boxes`) and
duplicates are merged with non-maximum suppression (IoU 0.45), which keeps the most confident box.

| Setup (4 test frames) | People found per frame | s/frame |
|---|---|---|
| yolo11n, no rotation, 960 px | 2, 4, 0, 3 | 0.05 |
| yolo11s, no rotation, 960 px | 4, 4, 2, 5 | 0.04 |
| yolo11n, 4 rotations | 9, 6, 4, 5 | 0.04 |
| **yolo11s, 4 rotations (chosen)** | **12, 11, 11, 11** | **0.065** |

The 4 rotations go to YOLO as one batch, so on the Mac GPU (MPS) they barely cost extra time.
`yolo11s` (small) finds clearly more people than `yolo11n` (nano). `--fast` in the CLI switches to
nano for slower laptops.

**Why not a fisheye-specific detector?** There are overhead-fisheye datasets (CEPDOF, WEPDTOF) and
rotation-aware detectors (e.g. RAPiD). Training or porting one would take much longer. The rotation
trick reuses an off-the-shelf model, needs no training data, and is easy to explain. With the
company's few hundred videos, fine-tuning YOLO on overhead data would be the next step.

### Remaining blind spot: the image centre
Directly under the camera a person is seen from straight above, just a head and shoulders. No
rotation helps there, so **detection is lost for ~10 frames every time someone crosses the centre**.
That's why the staff member's track breaks into ~20 pieces. Step 5 handles this.

## 2. Tracking: ByteTrack
Ultralytics normally runs its tracker inside `model.track()`. Our boxes come from merging four
rotated passes, so we call its `BYTETracker` class directly with a small adapter (`_Dets`).

ByteTrack matches detections to existing tracks using a Kalman motion model and box overlap. It
also uses low-confidence detections to keep tracks alive, which helps in crowded scenes.

| Parameter | Value | Why |
|---|---|---|
| track_high_thresh | 0.25 | same as the detector threshold |
| new_track_thresh | 0.3 | slightly stricter, so fewer one-frame ghost tracks |
| track_buffer | 50 | keeps a lost track for 2 s, for short occlusions |
| match_thresh | 0.8 | ultralytics default |

**Result:** 10.5 detections per frame, 159 tracks in total, 73 of them longer than 1 s.

### Known tracking errors (found while making the ground truth)
- **Breaks at the image centre:** see above.
- **ID swaps:** track 122 starts on one person and continues on the staff member. Track 385
  follows the staff member until frame 1208, then switches to a different dark-clothed person who
  walks past them.
- **Merged boxes:** two people walking close together sometimes get one box (around frame 950).

The tracker uses only motion and box overlap, not appearance, so people crossing each other can swap
IDs. A re-identification (ReID) model would reduce this, and is a possible next step.
