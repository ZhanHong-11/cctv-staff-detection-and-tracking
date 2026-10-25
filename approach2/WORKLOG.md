# Approach 2: work log

Built step by step with the user. Shared code for this approach lives in `src/staffid2/`
(`common.py`: paths + logging, `fisheye_view.py`: fisheye view geometry). Nothing is imported from approach 1;
code taken from it is copied.

## 2026-10-24: plain YOLO11s demo

- `scripts/yolo11_demo.py`: plain YOLO11s (no rotation, no tracking) on every frame, person boxes
  drawn into `outputs/yolo11_demo.mp4`.
- `../.venv/bin/python scripts/yolo11_demo.py`
- 1341 frames in 18.6 s on MPS (72 fps). Persons per frame: median 3, max 9 (the office has 10+).
- Frame 700: 6 people found, all in the top half. Upside-down people at the bottom and people at the
  centre are missed: the fisheye problem seen for ourselves.
- Shared `models/` moved to the project top level (approach 1 updated to use it).

## 2026-10-24: fisheye unwarp pictures

- `scripts/fisheye_unwarp.py`: cylindrical strip and multiple rectification (4 virtual perspective
  cameras), later quadrants (per-quadrant rotation) and approach 1's whole-frame rotations.
  Outputs in `outputs/fisheye_unwarp/frame{700,1000}_*.jpg`.
- `../.venv/bin/python scripts/fisheye_unwarp.py`
- The fisheye circle is larger than the frame (corners show the lens rim, not black), so centre and
  radius cannot be measured from a border; defaults are image centre and half diagonal (600 px), flags.
- Lens model is a guess: equidistant, 180 deg.
- 45-degree quadrant rotation needs a bigger canvas: about half of each rotated quadrant is black.

## 2026-10-25: compare view methods with YOLO

- Geometry moved to `src/staffid2/fisheye_view.py` (view makers with precomputed tables + mapping boxes back);
  `fisheye_unwarp.py` now uses it.
- `scripts/compare_methods.py`: 7 methods (plain, rotations, quad_axis, quad_diag, quad_axis_2x,
  rectified, cylindrical) x 50 frames spread over the video; YOLO at native view size; boxes mapped
  back, NMS 0.45; counts by ring, nested pairs, staff hit vs answer key, timings.
- `../.venv/bin/python scripts/compare_methods.py`
- Mapping back checked: exact (0.0 pixel diff) for 90-degree rotations/axis quadrants, 6–8 for
  resampled views (random pairs ~80).
- Results (MPS, mean per frame): rotations 10.4 people / 63 ms; quad_axis 8.4 / 25 ms;
  quad_diag 15.2 / 40 ms (inflated: loose boxes, 2.7 nested pairs); rectified 14.6 / 45 ms (tight
  boxes); quad_axis_2x 8.9 / 65 ms (enlarging barely helps); cylindrical 1.1 / 84 ms (fails);
  plain 2.8 / 13 ms. Centre blind spot remains for all.
- Write-up: `docs/compare_methods.md`, figures `docs/figures/compare_frame{700,1000}.jpg`.
- Problem: pandas `to_markdown` needs `tabulate` (not installed); wrote a small table formatter instead.
- Not done: CPU timing, per-person answer key (counts are not accuracy).
- Next: user to choose the method (candidates: rectified, quad_axis).

## 2026-10-25: rectified demo on the whole video

- `scripts/yolo11_rectified_demo.py`: like `yolo11_demo.py`, but YOLO11s runs on 4 rectified views
  (45 deg tilt, 90 deg FOV, 764 px = native detail); boxes mapped back to the original frame, NMS 0.45,
  drawn on the original video (box colour = view that found it). `--show-views` also writes the views.
- `../.venv/bin/python scripts/yolo11_rectified_demo.py` -> `outputs/yolo11_rectified_demo.mp4`
- 1341 frames in 62 s on MPS (21.6 fps overall). Per frame: views 3.0 ms, YOLO 42.2 ms, map back +
  merge 0.38 ms. Persons per frame: median 15, mean 14.8, max 23 (plain YOLO: median 3).
- Frame 700: 15 people (plain: 6), incl. the walking person and the bottom-left desks. Still missed:
  people at the far-left desks (x < ~180) and the image centre.
- Refactor: YOLO + map back + NMS (`detect`, `pick_device`, `yolo_size`) moved to
  `src/staffid2/detect.py`, `open_writer` to `common.py`, `native_view_size` to `fisheye_view.py`;
  `compare_methods.py` and `yolo11_demo.py` now import them (smoke-tested).
- Renamed `src/staffid2/views.py` -> `src/staffid2/fisheye_view.py` (user's request); imports and docs updated.

## 2026-10-25: answer key step A, rectified detection + ByteTrack

- Plan agreed with the user: build our own answer key by tracker-assisted labelling. The user clicks
  the staff member, the tool follows the ByteTrack track; gaps up to ~2–3 s are interpolated
  (flag in seconds), longer gaps are re-marked by the user. x,y = box centre.
- `src/staffid2/track.py`: ByteTrack wrapper copied from approach 1 (same parameters, track_buffer 50).
- `scripts/track_rectified.py`: rectified detection (YOLO conf 0.1 so ByteTrack's weak-match stage
  gets boxes) + ByteTrack -> `data/track_rectified/tracks.csv`, `track_summary.csv`,
  `outputs/track_rectified.mp4`.
- `../.venv/bin/python scripts/track_rectified.py`
- 1341 frames in 63 s (21 fps, MPS). 24.4 detections and 15.6 tracked boxes per frame.
  223 tracks: median 22 frames; 108 last >= 1 s; seated people often keep one ID for the whole
  video (tracks 3, 7: 1337 and 1333 frames).
- Staff member (checked against approach 1's draft key, certain frames only, rough guide): a track
  box covers the staff position in 93% of frames, but the staff member gets **24 different IDs
  in 37 runs** between frames 275 and 1251: while walking, the ID changes every ~1–2 s. Gaps
  between consecutive IDs are mostly a few frames.
- Consequence for step B: re-clicking at every ID change would mean ~30+ clicks; better to have the
  tool suggest the track that starts nearest to where the last one ended, for the user to confirm.

## 2026-10-25: answer key step B, click tool

- `src/staffid2/labeling.py`: labelling logic without UI. The user's decisions are a replayable list
  of actions (track / manual / cut / absent); `build_labels` turns them into one row per frame
  (later actions win), interpolates gaps <= max gap, leaves longer gaps "unlabelled";
  `suggest` proposes the track that continues the staff member after a loss; `jumps` flags
  possible ID swaps.
- Suggestion design, found by simulating a user session against approach 1's draft key:
  1. Only tracks that START after the loss are candidates (a fixed-distance version picked an
     already existing seated person's track at frame ~365).
  2. Allowed distance grows with the gap: 100 px + 30 px/frame (the staff member crossed the
     centre at ~28 px/frame: 285 px in 10 frames, 353 -> 364).
  3. Score = distance/allowed + 3 x gap/max_gap + speed difference/30. Gap weight 1/2/3 gave
     >100 px errors on 216/182/151 frames in the simulation; 3 chosen (tuned on the draft key, so
     only a default). Track 175 (static box) vs 181 (staff): 181 is now first.
  4. Remaining errors come from straight-line interpolation across detours (staff walked to the
     edge and back inside a gap). Hence the `p` key: play the gap with the would-be line drawn.
- `scripts/label_staff.py`: OpenCV window; keys in the docstring. Saves every action to
  `ground_truth/staff_labels.json` (resumable) and `ground_truth/ground_truth.csv`
  (frame, staff_present [1 / 0 / -1 unknown], x, y, box, source, track_id); on quit logs a
  summary (unlabelled / interp / absent ranges, jumps) and writes `outputs/staff_labels_review.mp4`.
- Tested headless (click, accept, undo, absent, save, resume, summary, review video) in the
  scratchpad. The window and keys themselves are untested: needs the user.
- `../.venv/bin/python scripts/label_staff.py --start 0`

## 2026-10-25: shared drawing code

- `src/staffid2/drawing.py`: `draw_box`, `draw_header`, `id_colour`, `label`, `mosaic` (the last two
  moved from `fisheye_view.py`, which now holds geometry only) and `table_video`: draw any table
  with frame + x1..y2 columns onto the original video (colour / text per row, header per frame).
- `track_rectified.py` now saves `tracks.csv` first and draws `outputs/track_rectified.mp4` from it
  with `table_video` (`--no-video` to skip). The CSV is the product; the video is a picture of it.
- `label_staff.py`: review video = `table_video` on the answer key; review videos are numbered
  automatically (`staff_labels_review_1.mp4`, `_2`, ...) so earlier ones are kept.
- `yolo11_demo.py`, `yolo11_rectified_demo.py`, `compare_methods.py`, `fisheye_unwarp.py` use the
  shared helpers.
- Checked: track video redrawn from the existing tracks.csv (1341 frames in 2.2 s, same IDs; the
  tracker was NOT re-run so IDs stay valid for labelling); label tool headless test incl. review
  video; fisheye_unwarp and a 3-frame compare run; all scripts import.

## 2026-10-25: label tool, range mode and gap checks

- User's first labelling session (21 actions) found two bad fills in the review: 32–34 s and 38–41 s.
  Cause: frames 819–846 and 989–1031 were interpolated *inside one track* (ByteTrack kept the ID
  alive and re-matched later), so the tool never stopped there and the straight line left the person.
- Changes (user chose 1–3 of the 4 proposals):
  1. Range mode `--from/--to` (s) or `--from-frame/--to-frame`: navigation limited to the range and
     every action is scoped to it. `build_labels` now replays session by session; a scoped session
     clears and re-fills interpolation inside its scope only, so labels outside never change.
     Undo only removes actions of the current session.
  2. `g` also stops at filled runs >= `--check-interp-s` (0.4 s); `p` plays them, Enter marks them ok
     (`ok` action, no effect on labels).
  3. A click with no track box under it stores a box the size of the nearest track box (not a point).
- Numbered versions: each tool run is session N; saves `ground_truth/ground_truth_N.csv` (+ latest
  `ground_truth.csv`) and `outputs/staff_labels_review_N.mp4` (review covers only the range).
  First session's files renamed to `_1`; its actions tagged `session: 1`.
- Tests (scratchpad copy of the user's labels): replay of the 21 actions identical to the saved CSV;
  scoped session 800–850 with clicks: all frames outside unchanged; scoped 950–1025 where the bad
  fill runs past the range (989–1031): frames 1026–1031 unchanged; replay of saved JSON == saved CSV.
- Note: a range that cuts through a bad fill only fixes the part inside it; choose ranges that
  cover the whole fill (e.g. 38–41.5 s for 989–1031).

## 2026-10-25: answer-key video

- Diagnosis of the two bad fills (user asked whether YOLO missed the person): 32–34 s = a real miss
  for 8 frames (819–826, image centre) plus the person tracked as new ID 362 from 827 while old ID
  270 stayed alive and re-matched at 847; 38–41 s = no miss at all: the person was track 452
  (990–1032) while the chain followed a 3-box duplicate ID 448 (988, 1032, 1033). In both cases the
  straight line filled the old ID's internal gap. Possible later improvement (not built): at a long
  internal fill, suggest another track with boxes along the gap.
- User already ran session 2 (800–850, fixed: clicks 819–826 + track 362 from 827) and started
  session 3 (950–1038).
- `scripts/draw_ground_truth.py`: `table_video` on `ground_truth/ground_truth.csv` (or any
  `ground_truth_N.csv`) -> `outputs/ground_truth.mp4`; colour by source, header with frame, time,
  source, x,y. `SOURCE_COLOR` moved to `drawing.py` (shared with the label tool).
- `../.venv/bin/python scripts/draw_ground_truth.py` -> 1341 frames in 1.9 s. Current key:
  track 1167, interp 165, manual 9 frames. Frame 823 checked: clicked box sits on the person.

## 2026-10-25: answer key finished, everything committed

- Session 3 (range 950–1038): click at 989, then track 452 from 990, replacing the bad fill 989–1031.
- The user watched `outputs/ground_truth.mp4` and confirmed the answer key is correct.
  Final `ground_truth/ground_truth.csv`: present in all 1341 frames; sources track 1167,
  interp 165, manual 9; 68 actions over 3 sessions.
- `detect.py`: `View` import made type-only, so plain detection does not need the fisheye code.
- `data/track_rectified/tracks.csv` + `track_summary.csv` force-added to git (data/ is ignored):
  the labels refer to these track IDs; re-running the tracker may renumber them.
- Committed in stages: plain YOLO -> fisheye views + comparison -> rectified demo -> tracking ->
  labelling tool -> answer key -> docs.
- Next: step 3, tag recognition (first look at tag visibility in staff crops from the answer key).

## 2026-10-25: step 3a started: staff crops and tag labelling tool

Plan agreed with the user for step 3 (tag recognition):
- Detect the tag only, inside person crops (people come from step 2); x,y output stays the person.
- Two detectors to compare: OpenCV NCC (template bank) and a small classifier on person crops.
  Full-frame YOLO fine-tuning rejected (tag ~13x7 px is about one cell at YOLO's finest stride).
- Training data / templates come ONLY from the brief's tag photo + a drawn tag, augmented
  (resolution, blur, compression, perspective, rotation, lighting, shirt/background, occlusion).
  Other people's crops = negatives / backgrounds (safe: no tag). The tags labelled in the video
  are used only to measure the real tag (to set augmentation ranges) and as the test set.
- Sightings (continuous ranges where the tag shows) are counted separately from frames: repeated
  corridor crossings give near-duplicate frames. Results are reported per frame and per sighting.
- First look (2 frames, scratchpad): on the black hoodie the tag is a bright ~17x9 px rectangle
  with 3 dips (bars); on the white shirt there is no bright rectangle at all, only the dark bars
  (~10x5 px). Shirt colour changes what the tag looks like, so both must be in the augmentation.

Built:
- `src/staffid2/person_crop.py`: `Lens.for_frame` (same lens guess as detection), `aim` (azimuth,
  tilt, fov, size computed from the box alone: tilt = distance from centre / f), `person_view`
  (rectified crop as a `View`), `original_crop`, `to_original_points` (sub-pixel crop -> frame).
- `scripts/staff_crops.py`: every 5th frame, staff crop from the answer key + up to 3 negatives
  (other tracks, skipping boxes with IoU > 0.3 with the staff box = duplicates of the staff).
  `../.venv/bin/python scripts/staff_crops.py` -> 269 staff crops (track 235, interp 32, manual 2),
  804 negatives from 115 tracks, 1.9 s. Staff crop 72-273 px (median 178), tilt 10-59 deg.
  Outputs: `data/tag_crops/` (PNGs + crops.csv), `outputs/tag_crops/staff_sheet_01..07.jpg`,
  `compare_original_rectified.jpg`.
- `scripts/label_tag.py` (user tool): rectified staff crop enlarged with pixel grid and zoom
  (x1-x4), s/e mark sighting ranges, drag a box around the tag on a few frames per sighting,
  ijkl/IJKL nudge by 1 px, undo, timeline strip. Right side shows the whole frame and the
  original crop with the tag box mapped back. Frames outside sightings = "not labelled".
  Saves `ground_truth/tag_labels.json`, `tag_boxes.csv` (crop box, w/h, aim, original-frame
  position), `tag_sightings.csv`.
- Tested headless (scratchpad output): drag at zoom x3 -> exact crop pixels; nudge + undo;
  save -> reload identical; box at frame 380 maps onto the tag in the original frame.

Next: the user labels the sightings with `label_tag.py`; then a measurement script (tag size,
squash, contrast, bar profile, coverage per sighting) and `docs/step3a_tag_survey.md`.

## 2026-10-25: tag labels done (user)

- User labelled with `label_tag.py`: 11 sightings (314-322, 334-345, 376-386, 426-436, 457-473,
  503-513, 542-559 white shirt; 809-821, 844-864, 890-905, 924-942 black hoodie), 158 frames,
  63 boxes (4-8 per sighting, none outside a sighting). Tag box w 5-26 px (median 16), h 5-21 px
  (median 11); the tag grows and gets less squashed as the person walks toward the centre.
- Not labelled, by the user's choice: seated period 0-274 (too blurry to see the tag) and after
  942 (same movement as the earlier hoodie passes, judged already covered). Frames 819-821 use the
  user's manual staff box from the answer key.
