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
models/                        shared YOLO weights (git-ignored) + approach 1's tag_templates/
approach1_groundtruth/         answer key built in approach 1 (staff_chain.json -> ground_truth_draft.csv)
approach1_template_matching/   finished: YOLO + rotations, ByteTrack, tag templates, track voting
approach2/                     in progress: rectified fisheye views + ByteTrack; has its own
                               USER-LABELLED answer key in approach2/ground_truth/ (use this one)
```
Inside each approach folder:
```
README.md, JOURNEY.md, WORKLOG.md, PLAN.md   how to run / what was tried / journal / plan
detect_staff.py      command-line entry point: --video <path> --out <dir>
src/<package>/       reusable pipeline code (the final pipeline only imports from here)
scripts/             one script per step: building, checking and inspecting on the sample video
docs/                one explanation file per step (how, why, results, figures)
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
  not "tight fit". (True for both keys; approach 2's key also has straight-line fills and user clicks.)
- **Answered by the user's labelling (approach 2 key):** the seated person on the left in frames
  0–274 IS the staff member, and the staff member is present in **all 1341 frames** (no absent frames).
- Walking across the centre the staff member moves up to ~28 px/frame (285 px in 10 frames, 353–364).
- With rectified views, ByteTrack still changes the walking staff member's ID every ~1–2 s, and it
  can keep an old ID "lost but alive" (track_buffer 50) while the person already has a new ID; the
  old ID later re-matches, leaving an internal gap (cause of two bad fills in the first labelling).

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
### Approach 2 (`approach2/`, in progress, built step by step WITH the user)
Working rules the user set (2026-10-24):
- Work only on the step the user names. Extras only if clearly beneficial, and **ask first**.
- Code from approach 1 may be reused but is **copied** into approach 2, never imported from it.
- Code shared by approach-2 scripts goes into `approach2/src/staffid2/` (not repeated per script).
- `scripts/` = things the user runs; `src/staffid2/` = reusable building blocks.
- Commit in logical sections (one commit per stage), only when the user asks.

Layout (no PLAN.md / JOURNEY.md / README.md yet; `WORKLOG.md` is the journal, read it first):
- `src/staffid2/`: `common.py` (paths `ROOT`, `PROJECT_ROOT`, `SAMPLE_VIDEO`, `MODELS_DIR`, logging,
  `open_writer`), `fisheye_view.py` (view makers + mapping boxes back), `detect.py` (YOLO on views
  + map back + NMS), `track.py` (ByteTrack wrapper, copied from approach 1), `labeling.py`
  (answer-key logic), `drawing.py` (`draw_box`, `table_video`: draw any CSV of boxes onto the video),
  `person_crop.py` (person-centred rectified crop: aim/tilt computed from the box),
  `tag_measure.py`, `tag_source.py`, `synth_tag.py`, `ncc.py`, `tag_score.py` (step 3).
- Step 3a scripts: `staff_crops.py` (staff + negative crops, contact sheets), `label_tag.py`
  (user tool: tag sighting ranges + tag boxes -> `ground_truth/tag_*.{json,csv}`), `tag_survey.py`
  (measurements via `src/staffid2/tag_measure.py` + figures).
- `scripts/`: `yolo11_demo.py` (plain YOLO), `fisheye_unwarp.py` (pictures of each view method),
  `compare_methods.py` (+ `docs/compare_methods.md`), `yolo11_rectified_demo.py`,
  `track_rectified.py` (-> `data/track_rectified/tracks.csv`, committed because the labels refer
  to its track IDs: **do not re-run it and overwrite that file**), `label_staff.py` (click tool),
  `draw_ground_truth.py` (-> `outputs/ground_truth.mp4`).

Done:
- [x] Plain YOLO11s: median 3 people/frame; misses upside-down (bottom) and centre people.
- [x] View methods compared (50 frames, MPS): rectified 4 views (45° tilt, 90° FOV, 764 px = native
  detail) 14.6 people/frame, 45 ms, tight boxes -> **chosen**. Rotations (approach 1) 10.4 / 63 ms;
  quad_axis 8.4 / 25 ms (cheap fallback, no lens model); quad_diag 15.2 but inflated loose boxes;
  enlarging 2x barely helps; cylindrical fails. Centre blind spot remains for all.
  Lens model is a guess (equidistant, 180°, centre = image centre, radius = half diagonal).
- [x] Rectified detection + ByteTrack on the whole video: 21 fps on MPS, 223 tracks; seated people
  keep one ID, the walking staff member gets ~24 IDs.
- [x] **Answer key labelled and checked by the user**: `approach2/ground_truth/ground_truth.csv`
  (`frame, staff_present [1/0/-1 unknown], x, y, x1..y2, source [track/interp/manual/absent/unlabelled],
  track_id`; x,y = box centre). 3 sessions, numbered copies `ground_truth_N.csv`; actions in
  `staff_labels.json`. Final: track 1167, interp 165, manual 9 frames; present 0–1340.
  Fill rule: gaps <= 2.5 s straight-line interpolated; the tool makes the user check fills >= 0.4 s.

Step 3 tag recognition (see WORKLOG 2026-10-25/26): NCC only (the user dropped the classifier).
Templates come only from the brief's tag photo; video tags labelled by the user are for measuring
and testing only.
- [x] 3a user labelled 11 sightings / 63 tag boxes; measurements in `docs/step3a_tag_survey.md`.
- [x] 3b tag cut automatically from the brief's photo (`tag_source.py`, user confirmed the box),
  generator `synth_tag.py` checked against real tags (`docs/step3b_synth_tags.md`).
- [x] 3c NCC detector `ncc.py` (`no_mid`: light + dark looks, score = tag NCC - "no bars" block
  NCC); hit threshold 0.037 finds 16 % of visible-tag frames, 9/11 sightings, 1 % false hits
  (`docs/step3c_ncc.md`). Open risk: no mid-tone (grey shirt) templates.
- [x] Scores for all tracked boxes (`score_tracks.py` -> `data/tag_scores/`), viewer
  `tag_score_viewer.py` (user tool), live scorer `score_tracks_live.py` (first version, not fast
  enough for 25 fps yet: ~10 ms per crop per core).
- Scripts added: `synth_tags.py`, `ncc_eval.py`, `score_tracks.py`, `score_tracks_live.py`, `score_charts.py`,
  `tag_score_viewer.py`; src: `tag_source.py`, `synth_tag.py`, `ncc.py`, `tag_score.py`; paths
  `BRIEF_PDF`, `TRACKS_CSV`, `ANSWER_KEY`, `TAG_BOXES`, `TAG_SIGHTINGS`, `TAG_SCORES` in `common.py`.

Step 4 linking: a first version (anchors = score > 0.065, then following forward AND backward,
93.5 % of frames right) is in `approach2/review_pending/` and was NOT accepted. **The user wants
live logic**: no staff until the tag is seen (strong hit), then follow the track; when the track
is lost, look nearby (starts soon, close, similar speed); if nothing is found, back to searching
for the tag. Full agreed plan + decisions: WORKLOG "2026-10-26: step 4 plan agreed" (confirmation
0.065 anywhere, normal hit 0.037 only in context, focus = colour only, overlap/centre/standstill
handlers only where needed and only for the green target, several targets allowed, gaps filled
only in the saved CSV at the end, 3-number scoring). Score charts: real tag = rare tall spikes on a
moving person; false hits = many medium bumps on still people just under 0.065
(`docs/figures/step3c_score_over_time.png`).
New step 4 work is done in `approach2/review_pending/` (not committed) until the user accepts it.

Next (agreed order): 4.1 rules doc -> 4.2 basic follower (`follow.py`, frame by frame on the saved
scores) -> 4.3 follow layer in the viewer -> 4.4 score it -> 4.5 special cases where needed ->
4.6 robustness -> 5 evaluation (also approach 1 on the key) -> 6 `detect_staff.py` CLI (live
speed-ups), docs.

### Approach 1 (`approach1_template_matching/`, finished)
- [x] venv created, video and PDF analyzed, PLAN.md written
- [x] Step 1 ground truth **draft** (`approach1_groundtruth/`, needs checking by the user)
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
- The approach 1 draft key was never checked; the user-labelled approach 2 key answers its open
  questions (seated person 0–274 is staff; staff present until the last frame, not leaving at ~1257).
- Verified after the move into `approach1_template_matching/` (2026-10-24): `scripts/step5_vote.py`
  gives the same result as before.
- Step 9 (the final 1–2 page write-up) is written **by the user**. Claude's job is to explain
  everything thoroughly in `docs/` and `WORKLOG.md` so the user has the material to write it.
