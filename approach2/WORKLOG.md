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

## 2026-10-25: step 3a tag survey (measurements + doc)

- Committed first (user asked): tools, user tag labels, docs as 3 commits (81ebe6a, 104ab71, f5d6919).
- `src/staffid2/tag_measure.py`: shirt band vs inside, best-row bar profile + dip count, edge
  sharpness, tag rotation/length/thickness from 2nd moments of pixels differing from the shirt.
  Shared so 3b can measure synthetic tags the same way.
- `scripts/tag_survey.py` -> `outputs/tag_survey/{tag_measurements,sightings,ranges}.csv` and
  `docs/figures/step3a_{gallery,contrast,profiles,size_vs_tilt,coverage}.png` (1 s).
- Fixed after the first run: row-averaged profile smeared rotated bars (hoodie median 0 dips);
  gradient bar angle gave +-85 deg (outer edges dominate). Now best single row + moment-based
  angle; box w/h replaced by rotation-free length/thickness.
- Results: tag length 5.3-19.2 px (median 12.8), thickness 3.5-10.4, ratio 0.30-0.86, rotation
  -38..+59 deg even in rectified crops. Light shirt: card invisible (tag - shirt -17..+27), only
  bars show; dark hoodie: shirt clipped to 0, card +60..+139. Bars resolved in 48/63 boxes;
  unresolved = small (<10 px), rotated (>30 deg) or fast (28-35 px/frame). Length falls with
  tilt (19 px at 20 deg -> 5-8 px at 60 deg). Coverage: central corridor + one bottom-left spot;
  repeat crossings at the same places.
- Doc: `docs/step3a_tag_survey.md` incl. suggested 3b augmentation ranges (length 4-24 px,
  ratio 0.25-0.9, rotation +-70 deg, light/dark/mid shirts, card-invisible case, blur matched by
  measuring synthetic tags).
- Next: 3b synthetic tag generator (from the brief's tag photo + a drawn tag).

## 2026-10-25/26: step 3b synthetic tags (built by Claude unattended; checked by the user 2026-10-26)

- Built overnight in `review_pending/` at the user's request, reviewed by the user, then moved here
  (2026-10-26). The user confirmed the automatically found tag box is the tag.
- `src/staffid2/tag_source.py`: the brief PDF's embedded JPEGs are read directly (no PDF
  library). Only image 0 (green-shirt photo, 218x233) is used; images 1-2 are frames of the
  sample video (would leak test data). Tag = white unsaturated blob, extended over rows/columns
  that are not green shirt (margin 8) -> box 36x22 px. Layout measured automatically
  (`outputs/tag_source/layout.json`): aspect 1.64, black band top 23 %, 3 bars hanging to 64 %
  of the height, outline 0.045. Explains 3a's two looks (dark comb on a light shirt where the
  card blends in; white comb on the hoodie where band and bars blend in).
  `photo_albedo.png` = the photo's tag as 0 (ink) .. 1 (white); `drawn_albedo.png` = a clean tag
  drawn from the layout (used for the templates).
- `src/staffid2/synth_tag.py`: renderer (8x supersampling, blur, motion blur, noise, JPEG);
  ranges from 3a's `ranges.csv`, widened. 600 random tags pasted on other people's chests
  (`data/tag_crops/neg`), look chosen first then a torso of matching brightness, measured with
  `tag_measure` and compared with the 63 real tags. Four range fixes (thinner ratio, darker hoodie
  card, ink = 5-50 % of card, look-first torso). Real values inside the synthetic 1-99 % range for
  every measure (card-minus-shirt 92 %); synthetic still a bit rounder (median 0.63 vs 0.49).
  The 600 are only a check of the generator (the classifier they were meant for was dropped).
- `../.venv/bin/python scripts/synth_tags.py` (3 s) -> `outputs/tag_source/`,
  `outputs/synth_tags/{synth_measurements,compare}.csv`, `docs/figures/step3b_{source,gallery,compare}.png`.
  Same seed (0) every run. Doc: `docs/step3b_synth_tags.md`.

## 2026-10-25/26: step 3c NCC tag detector (built by Claude unattended; reviewed by the user)

- `src/staffid2/ncc.py`: 140 templates (drawn tag; lengths 6/8/11/14/18 px, ratios 0.3/0.5,
  angles -60..60 step 20, looks light + dark), made in memory each run (no randomness, 0.04 s).
  Search only the chest area of the upright crop (x 25-80 %, y 10-70 %), OpenCV TM_CCOEFF_NORMED,
  windows with std < 6 ignored. Score ("verify") = NCC of the best tag template - best NCC of its
  two "no bars" blocks (gaps filled dark / white) at the same spot. Range in theory -2..2; in
  practice almost all boxes are -0.13..0.07.
- `../.venv/bin/python scripts/ncc_eval.py` (~3 min): 9 configurations on 158 sighting frames
  vs 3859 crops of other people; threshold set on even track IDs (2 % pass), reported on odd.
  Chosen `no_mid`: threshold 0.037, finds 15.8 % of visible-tag frames, 9/11 sightings, match on
  the user's tag box 10/63, false hits 1.0 %. raw NCC 6.3 % (a white logo on a black polo scored
  0.97); mid-tone templates made 57/69 false hits (dropped; risk for a grey-shirted staff member).
  Outputs `outputs/ncc_eval/`, `docs/figures/step3c_{scores,examples}.png`. Doc: `docs/step3c_ncc.md`.
- Correction: the earlier note "1.6 ms per crop per core" was wall time with 7 processes; on one
  core matching takes ~10 ms per crop (~160 ms per frame of 16 people).

## 2026-10-26: tag scores for every tracked box + viewer + live scoring (first version)

- `src/staffid2/tag_score.py`: `gray_crops` + `score_crops` (batch, one process per core) and
  `FrameScorer` (live: one frame's boxes at a time, nothing from other frames; optional threads).
- `../.venv/bin/python scripts/score_tracks.py` (40 s, 7 processes) -> `data/tag_scores/ncc_track_scores.csv`:
  score for all 20 863 tracked boxes. Each score uses only its own frame; the whole video is done
  in one go only so the viewer can browse instantly. Over all boxes: other people median -0.003,
  max 0.065; staff max 0.107. Frames with a hit (>= 0.037): 292; with a score > 0.065: 11
  (385, 432, 468, 470, 471, 506, 507, 558, 1019, 1156, 1201), all on the staff member.
- `scripts/tag_score_viewer.py` (user tool): a/d frames, every tracked box coloured by score
  (green > 0.065, orange >= 0.037, grey, dark = negative), matched spot outlined, selected box's
  crop + "seen / tag / block dark / block white" with live NCC values, strips of the best and the
  staff box's score over the whole video, jumps to strong/hit frames. Does not decide or follow
  the staff member (the cyan box is the answer key, for comparison). Live recompute matches the
  CSV to 5e-5.
- `scripts/score_tracks_live.py`: frame-by-frame scoring with timing against the 25 fps budget;
  `../.venv/bin/python scripts/score_tracks_live.py --compare` (102 s). Scoring per frame (median 16
  boxes): 1 thread 153 ms (6.5 fps), **4 threads 67-75 ms (13-14 fps, default)**, 7 threads 119 ms
  (slower: Python parts of the matcher compete for the GIL). Over the 40 ms budget in 1335/1341
  frames; detection + tracking (~48 ms on MPS) come on top. Same scores as the batch run (max
  difference 5e-5). Not optimised yet (ideas: template sizes from the crop size, every 2nd-3rd frame,
  re-check seated people less often).
- Moved from `review_pending/` after the user's review: the code, scripts, docs and figures of
  3b, 3c, the scores and the viewer; re-run from here gave identical outputs. Still under review
  in `review_pending/`: step 4 linking (anchors + forward/backward following), which the user did
  not accept (it fills the staff track before the tag is seen; the user wants live logic).
- `scripts/score_charts.py` -> `docs/figures/step3c_score_over_time.png`, `step3c_score_zoom.png`:
  staff score over time vs two seated non-staff tracks (63: 47 hits, max 0.063; 292: 25 hits in
  66 frames, max 0.060). Real tag = rare tall 1-3 frame spikes on a moving person; false hits =
  many medium bumps on a still person, just below the 0.065 line. Over all tracks: staff hit runs
  1-2 frames (max 5), others' hits spread over many seconds. A "drop 1-2 frame spikes" filter
  would remove the real hits.

## 2026-10-26: step 4 plan agreed (user's live logic)

- First step 4 version (anchors + following backward and forward) not accepted: it fills in the
  staff member before the tag is seen. Kept in `review_pending/` only until replaced.
- Agreed design (user's idea, adjusted after the score charts):
  - Two thresholds: confirmation (0.065) anywhere = green target; normal hit (0.037) only counts
    in context (near a lost target, or on a suspect after a split). "Focus" = colour only at first.
  - Follow the green target's track; when lost, the generic continuation search (starts soon,
    close, similar speed); nothing found within a time limit -> back to searching.
  - Special losses, added only where the evaluation shows a need: (1) overlap / ID swap, only for
    the green target: after the split all boxes involved become suspects until a tag decides;
    (2) centre blind spot: accept a newborn track on the far side in the direction of travel;
    (3) standing still / changing clothes: hold the last known position.
  - Several green targets allowed; finding one never drops another.
- Decisions: (1) main CSV `frame,staff_present,x,y` = the most recently confirmed target; a second
  CSV lists every green target per frame. (2) Gaps are filled only in the saved CSV at the end
  (straight-line interpolation, or deduced from the surroundings, between two sightings of the
  same target); the live view is not filled. (3) Scored as: time to first confirmation,
  correctness when the output says "staff" (right person), coverage after the first confirmation.
  (4) Focus = colour only in the first version.
- New work happens in `review_pending/` until the user accepts it. Next: 4.1 rules doc (states,
  settings) for the user to check before any code.

## 2026-10-26: step 4.2-4.4 live follower, viewer, score (in `review_pending/`, under review)

- The user: "just go first and we do refinement later", with a viewer like the tag score viewer.
  Colours agreed: focus = orange (a hit in the last 1 s, colour only), suspect = yellow (4.5).
- `review_pending/src/follow.py` (Follower, fill_gaps, main_output, score),
  `review_pending/scripts/follow_staff.py` (run + score, ~1 s, 0.2 ms/frame for following),
  `review_pending/scripts/follow_viewer.py` (user tool: live / saved views, events, strips).
  Rules doc updated (`review_pending/docs/step4_rules.md`, additions marked 4.2); results in
  `review_pending/docs/step4_follow.md`.
- Result (live): first confirmation 385; right when it says staff 83.9 % (686/818); right after the
  first confirmation 71.8 % live, 81.7 % saved (end fill). 123 of the 132 wrong live frames are the
  ID swap at 867-1018 (4.5 case 1).
- Rules added while building (each re-scored, all generic but found on this video): switch an
  UNCERTAIN target to a better continuation of the same loss (53 -> 73 %); keep the earlier loss
  point after a brief guess, `trust_frames` = 10; a hit makes a target green again but only a
  confirmation (> 0.065) picks the main target, which stays main while LOST.
- Open: the swap; a stale wrong target stays green forever after a hit on a seated person; hits on
  guessed tracks of seated people. Next: the user reviews in the viewer, then 4.5.

## 2026-10-26: step 4.5 case 1, suspects after a loss (in `review_pending/`, under review)

- Why: at the ID swap (865) the staff member got old IDs of other people (142, 369); the
  continuation search only takes tracks born after the loss, so the right person was never a
  candidate, although the tag showed on them (hits at 890, 925, 928).
- Rule (user agreed to try): when a target is lost, everyone within 250 px of the loss point for
  5 s becomes a suspect (yellow), old tracks included; a hit on a suspect that moved >= 40 px in
  the last 25 frames moves the target there. Runs next to the movement guess, not instead of it.
- First hit wins: worse (live 81.6 % right when said): another walking person (track 384, never
  staff) hit 0.055 at 881, before the staff member's 0.052 at 890. Latest hit wins (window stays
  open until a confirmation or 5 s): **live 95.1 % right when said, 80.8 % after the first
  confirmation; saved 92.2 %** (4.2: 83.9 / 71.8 / 81.7). Wrong live frames 132 -> 40.
- Viewer: suspects drawn yellow, listed under the target. Docs: `review_pending/docs/step4_follow.md`
  (section 4.5), `step4_rules.md`, `tag_recognition_&_staff_tracking.md` (rule 10).
- Settings are first guesses and the variant was chosen on this one swap: optimistic for a new video.
- User's choice: while the suspect window is open, a hit-made green is shown light green (UNCERTAIN)
  because a later suspect hit can still move it; solid green only after a real confirmation or the
  window's end. Same CSV; solid green is now 196 frames, all on the right person.
- Checked ~1209 (staff box 553 overlapping 478 at IoU 0.22): the tracker kept the ID on the staff
  member, the follower stayed right; the 0.075 confirmation at 1201 had closed the window.

## 2026-10-26: step 4 accepted and moved out of `review_pending/`

- The user accepted step 4. Moved: `review_pending/src/follow.py` -> `src/staffid2/follow.py`;
  `follow_staff.py`, `follow_viewer.py` -> `scripts/`; `step4_rules.md`, `step4_follow.md`,
  `tag_recognition_&_staff_tracking.md` + figures -> `docs/`; outputs -> `outputs/follow/`; logs ->
  `logs/`. `review_pending/` removed. Re-run from the new place: `score.json` and `staff.csv`
  byte-identical to before the move.
- Next: the confirmation threshold (fitted on this video) and the `detect_staff.py` CLI (results in
  a `result/` folder, as the user asked).

## 2026-10-26: step 6 `staff_detector.py` (the command for a new video)

- Committed step 4 as `48cd1c7` (the user asked). `ground_truth/tag_labels.json` was left out: only
  its "saved" time changed (the user opened `label_tag.py`), no labels changed.
- Threshold check before building (scratchpad runs of `follow_staff.py --conf-thr`): 0.055 / 0.06 ->
  first "confirmation" at frame 45 on a wrong person, 43 / 40 % right; 0.065 ... 0.10 -> identical
  result (one good confirmation at 385 is enough); > 0.107 -> nothing reported. "Moving only" does
  not separate: the highest non-staff score (0.0648) is on a walking person. The user wants an
  adaptive threshold but it cannot be tested here: no change for now.
- New: `src/staffid2/pipeline.py` (`PersonTracks` = the detection + ByteTrack loop of
  `track_rectified.py` as reusable code; `tag_bank`), `drawing.dashed_rect` (moved from the follow
  viewer) and `drawing.draw_follow`, `staff_detector.py` (name chosen by the user) -> `result/<video>/`
  (the user wants results apart from `outputs/`; `result/**/*.mp4` git-ignored).
- `../.venv/bin/python staff_detector.py --answer-key ground_truth/ground_truth.csv`: 1341 frames in
  173 s (7.7 fps; people 55 ms, tags 73 ms, follow 0.4 ms per frame). Same tracks as `tracks.csv`
  and the same result as `follow_staff.py`: first confirmation 385, live 95.1 % / 80.8 %, saved 92.2 %.
- Fixed on the way: boxes kept as float32 moved 2 of 1911 tag scores (crop shifted by a hair);
  now float64 like the CSV, scores identical.
- `scripts/track_rectified.py` now calls `PersonTracks` (tested into the scratchpad: same 223 tracks).
- Doc: `docs/step6_staff_detector.md`. Next: the user tries it; adaptive threshold idea; evaluation (5).
