# Scripts

The 19 scripts in `approach2/scripts/`, grouped by purpose. They are run on the sample video to
build, check and inspect each step. Shared code lives in `src/staffid2/`; the command for a new
video is `approach2/staff_detector.py`, not a script here.

⭐ = user tool (interactive: you click, step or label in it).

## 1. Fisheye lens & YOLO experiments

| Script | Purpose |
|---|---|
| `yolo11_demo.py` | Plain YOLO11s on every frame, written to a video. Shows what an off-the-shelf model misses (upside-down people at the bottom, people in the centre). |
| `fisheye_unwarp.py` | Pictures of sample frames reshaped 4 ways (rotations, rectified views, quad, cylindrical) to compare them by eye. |
| `compare_methods.py` | Measures each view method on 50 frames: people found per frame, speed, box tightness. Chose the rectified 4 views. Results in `docs/compare_methods.md`. |
| `yolo11_rectified_demo.py` | Same as `yolo11_demo.py` but on the rectified views, so the two videos can be compared side by side. |

## 2. Staff labelling & ground-truth build-up

| Script | Purpose |
|---|---|
| `track_rectified.py` | **Step A**: rectified YOLO + ByteTrack over the whole video -> `data/track_rectified/tracks.csv`. The labels refer to its track IDs: **do not re-run it and overwrite that file**. |
| ⭐ `label_staff.py` | **Step B**: click tool. Click the staff member, it follows the track and suggests the next one when it ends. Writes `ground_truth/ground_truth.csv` (latest) + `ground_truth_N.csv` (one copy per session) and `staff_labels.json` (all actions). |
| `draw_ground_truth.py` | Draws the finished answer key onto the video (`outputs/ground_truth.mp4`), boxes coloured by source (track / interpolated / manual click). |

## 3. Tag synthesis (steps 3a–3b: learning what the tag looks like)

| Script | Purpose |
|---|---|
| `staff_crops.py` | **3a part 1**: person-centred rectified crops of the staff member and of other people (negatives), plus contact sheets. |
| ⭐ `label_tag.py` | **3a part 2**: mark the frame ranges where the tag is visible (sightings) and draw boxes around it -> `ground_truth/tag_*.{json,csv}`. |
| `tag_survey.py` | **3a part 3**: measures the labelled tags (size, squash, rotation, camera tilt, speed) to show how much they vary. Results in `docs/step3a_tag_survey.md`. |
| `synth_tags.py` | **3b**: cuts the tag out of the brief's photo, makes synthetic tag templates and checks them against the real tags. Results in `docs/step3b_synth_tags.md`. |

## 4. Tag recognition (step 3c: finding the tag with NCC)

| Script | Purpose |
|---|---|
| `ncc_eval.py` | Evaluates the NCC detector against the user's tag labels; source of the hit threshold and hit / false-hit rates (`docs/step3c_ncc.md`). |
| `score_tracks.py` | Tag score for every tracked box in every frame -> `data/tag_scores/`. Step 4 runs on these. |
| `score_tracks_live.py` | The same scoring done frame by frame like a live run, to measure the cost (~10 ms per crop per core, not yet fast enough for 25 fps). |
| `score_charts.py` | Charts of the tag score over time, staff vs others: why one strong hit works and counting hits does not. |
| ⭐ `tag_score_viewer.py` | Step through the video frame by frame with every box coloured by its tag score. |

## 5. Others: staff following (step 4)

| Script | Purpose |
|---|---|
| `follow_staff.py` | Runs the live follower on the saved tag scores (~1 s) and scores it against the answer key -> `outputs/follow/`. Results in `docs/step4_follow.md`. |
| ⭐ `follow_viewer.py` | Shows what the follower decided frame by frame: target, suspects, events, live and saved views. |

## Order they were run in

1 (choose a view method) -> 2 (tracks, then the answer key) -> 3 (tag labels, survey, templates)
-> 4 (NCC detector, scores for all tracks) -> 5 (follower). Step 6, `staff_detector.py`, runs
detection + tracking + scoring + following in one live pass on any video.
