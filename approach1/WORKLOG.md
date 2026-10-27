# Work log

## 2026-10-23: setup and Step 1 (exploration)

**Done**
- Created `.venv` (Python 3.14.7), installed `opencv-python` and `numpy`.
- Wrote `CLAUDE.md` (project rules), `PLAN.md` (plan) and `.gitignore` (excludes `data/`, `logs/`, `*.pt`, videos in `outputs/`).
- `src/staffid/common.py`: logging to the console and `logs/`, and a frame iterator.
- `src/staffid/motion.py`: MOG2 background-subtraction detector for moving people.
- `scripts/step1_explore.py`: motion boxes for every frame, and zoomed crops.
- `scripts/step1_figures.py`: exploration figures in `docs/figures/`.

**Commands**
```
.venv/bin/python scripts/step1_explore.py   # 1025/1341 frames with motion, 1720 boxes, about 4 s
.venv/bin/python scripts/step1_figures.py
```

**Key findings** (details in `docs/step1_explore.md`)
- The staff member appears in two outfits: a white shirt (about frames 330–640) and a black hoodie (about frames 690–1200). The tag is visible on both.
- The tag is about 13×7 px with bars 1–2 px wide, so it's often unreadable in a single frame. Combining evidence per person over time is needed.
- Background subtraction misses people who stand still (frames 660–810) and sometimes merges people, so YOLO is needed.
- Exposure dip at frames 1300–1305 (frame difference 99.6 against a median of 1.4).

**Problems**
- `pip install ultralytics lap` was **blocked by Claude Code's auto-mode permission check**, even though the user had approved it in chat. The user has to allow it, or run the install themselves.

**Next**
- Install `ultralytics` and `lap`, then Steps 2 and 3 (YOLO detection and tracking), then draft ground truth by hand-checking tracks.
- In the meantime: Step 4A tag detector prototype using OpenCV only.

## 2026-10-23: installs, Steps 2–8

**Installs:** the user ran `.venv/bin/pip install -r requirements.txt` (ultralytics 8.4.184, torch 2.14.1, etc.); versions are pinned in `requirements.txt`. Python 3.14 works and MPS (Mac GPU) is available. Ultralytics analytics are turned off in code (`SETTINGS.update({"sync": False})`). The YOLO weights downloaded to `models/` (git-ignored).

**Steps 2–3: detection and tracking** (`docs/step2_3_detect_track.md`)
- Plain YOLO found 0–5 of ~20 people per frame (people look upside down at the bottom of the fisheye image). Detecting on 4 rotations raised that to 11–12. Chosen: yolo11s, 960 px, 4 rotations + NMS, 0.065 s/frame.
- ByteTrack via an adapter: 159 tracks. Detection is lost at the image centre (seen from straight above), so the staff member's track breaks into ~20 pieces.

**Step 1 (continued): ground-truth draft** (`docs/step1_explore.md`, `approach1_groundtruth/`)
- Built from the tracks, checked by eye. Findings: the staff member puts the hoodie on at ~633–653; a second dark-clothed person (not staff) is present and the tracker's ID swaps onto them at ~1211; the seated person in frames 0–274 is probably the staff member (uncertain).
- `pos_certain` was added after frame 575 showed that long interpolation gaps are guesses.

**Step 4A: tag detection** (`docs/step4_tag_detection.md`)
- Upright crops (rotation from fisheye geometry) + 9 real tag templates.
- Matching alone: AUC 0.557 (laptop edges matched). Added a surround check (bright patch on dark / dark bars on light) and a person-sized chest window: 0% hits on other people, 18% on staff boxes.
- Speed: 236 → 13.5 ms/box. **Note:** the first full run would have taken ~50 min; the user asked whether it was stuck, so I stopped it and reduced the template variants, upscale, and frames scored.

**Steps 5–7: linking, voting, evaluation** (`docs/step5_vote.md`)
- First linking rule (distance only): median position error 250 px (chains mixed people).
- Calibrated on the true hand-offs (gap 7–14 frames, jump 150–390 px) → direction-consistent jump links. 2 staff chains; certain frames: precision 1.0, recall 0.897, 94.9% within 25 px.
- Ablation: linking + gap filling add ~27 recall points.

**Step 8: CLI + robustness** (`docs/step8_robustness.md`, `README.md`)
- `detect_staff.py`: one command; 145 s for the sample video.
- Robustness on mirrored / rotated / shrunk / darkened copies found long bad chains (one ID swap spreading the label to the wrong person for ~1000 frames). Added `min_hit_rate = 0.03`: identity accuracy within 25 px is now 93–98% on all variants; recall on the variants is 0.56–0.67 (the white-shirt part is the weakest).

**Commands**
```
python scripts/step2_3_detect_track.py
python scripts/step1_build_ground_truth.py
python scripts/step4_make_templates.py && python scripts/step4_score_tags.py
python scripts/step5_vote.py            # --no-link / --no-fill for ablations
python scripts/step8_robustness.py      # ~9 min, caches tracks/scores
python scripts/step8_revote.py          # compare voting rules in seconds
python detect_staff.py --video "AI Test Candidates.mp4" --gt approach1_groundtruth/ground_truth_draft.csv
```

**Open items**
- The user needs to check the ground-truth draft (seated period 0–274, exit at ~1257).
- Optional 4B: trained tag classifier (the biggest expected gain in robustness).
- No test on a video without staff.

## 2026-10-23/24: wrap-up

- Annotated video switched to H.264 (`avc1`, fallback `mp4v`): plays in QuickTime, and the file shrank from 40 MB to 10 MB.
- `JOURNEY.md` added: what was tried, what failed, and how the strategy changed at each stage.
- Project reorganised (in a side session): approach 1 moved into `approach1_template_matching/`. The sample video, `approach1_groundtruth/`, `requirements.txt` and `.venv` are shared at the project root, and `src/staffid/common.py` gained `PROJECT_ROOT`, `SAMPLE_VIDEO` and `GT_DIR`. Checked after the move: `scripts/step5_vote.py` gives the same result (2 staff chains, recall 0.897).
- Approach 2 hasn't started; the options are listed in the root `CLAUDE.md`.

## 2026-10-27: moved into `approach1/`
- `approach1_template_matching/` -> `approach1/`, `approach1_groundtruth/` -> `approach1/groundtruth/`
  (the user wanted all of approach 1 in one folder). `common.GT_DIR` is now `ROOT / "groundtruth"`;
  README, JOURNEY, docs, scripts' docstrings and approach 2's `compare_methods.py` updated.
- Checked: `../.venv/bin/python scripts/step5_vote.py` -> 2 staff chains, recall 0.897 on certain
  frames, 95% of positions within 25 px: same as before the move.
- The user decided approach 1 is no longer used: it is the first experiment, kept for reference
  only. Notes added to README.md, JOURNEY.md, PLAN.md, the top-level README.md and CLAUDE.md.
