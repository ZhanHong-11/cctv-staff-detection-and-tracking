# Step 6: `staff_detector.py`, the command for a new video

```
cd approach2
../.venv/bin/python staff_detector.py --video <file.mp4>                 # -> result/<video name>/
../.venv/bin/python staff_detector.py --video <file.mp4> --no-video      # CSVs only
../.venv/bin/python staff_detector.py --answer-key ground_truth/ground_truth.csv   # sample, with score
```

## What it does

One pass over the video, frame by frame, the way a live camera would run it (frame t never uses
later frames):

| Stage | Code | Per frame (median, sample, Mac GPU) |
|---|---|---|
| 1. people: 4 rectified views → YOLO11s → boxes mapped back + merged → ByteTrack IDs | `src/staffid2/pipeline.py` (`PersonTracks`) | 55 ms |
| 2. tag score of every tracked person (140 NCC templates from the brief's tag photo, 4 threads) | `tag_score.FrameScorer`, `pipeline.tag_bank` | 73 ms |
| 3. following (confirmation, track, movement guess, suspects) | `follow.Follower` | 0.4 ms |
| after the last frame: fill gaps ≤ 2.5 s, write CSVs, draw the video (second pass) | `follow.fill_gaps`, `drawing.draw_follow` | ~3 s in total |

Whole sample video (1341 frames): **173 s, 7.7 fps**. Not real time (25 fps needs ≤ 40 ms per
frame); the tag scoring is the slow part (ideas in `step3c_ncc.md`).

The rectified views are made fresh for every frame of the new video from the frame size alone
(lens model: equidistant 180°, centre = image centre, radius = half the diagonal: a guess, the
same as for the sample). Nothing from the sample video is used except the settings.

## Outputs (`result/<video name>/`)

| File | Content |
|---|---|
| `staff.csv` | **the answer**: `frame,staff_present,x,y` (gaps ≤ 2.5 s filled at the end) |
| `staff_live.csv` | the same as decided live, no filling |
| `detail.csv` | `staff.csv` + target, state, track ID, source (live / interp / held) |
| `targets.csv`, `events.csv` | every target per frame; what the follower did and why |
| `tracks_scored.csv` | every tracked person box with its tag score |
| `run.json` | settings, timings, counts, first confirmation (+ score with `--answer-key`) |
| `staff.mp4` | the video: STAFF label on the reported person, colours as in the follow viewer (git-ignored) |

Every setting is a flag (`--help`): detection (`--conf`, `--n-views`, ...), ByteTrack
(`--track-buffer`, ...), following (`--conf-thr`, `--hit-thr`, `--suspect-radius`, ...).

## Checked on the sample

- Tracking: identical to `data/track_rectified/tracks.csv` (same 20 863 boxes and IDs; at most
  0.1 px rounding on a few boxes, because the boxes are now rounded from float64, not float32).
  A first version kept float32 boxes and 2 of 1911 tag scores moved by up to 0.04 (the crop shifted
  by a hair); fixed.
- Result: **the same as `scripts/follow_staff.py` on the saved scores**: first confirmation 385;
  live 95.1 % right when it says staff, 80.8 % right after the first confirmation; saved 92.2 %.
- `scripts/track_rectified.py` now uses the same `PersonTracks` loop (tested into the scratchpad:
  same 223 tracks, 20 863 boxes; the real `tracks.csv` was not overwritten).

## Limits on a new video

- **Confirmation threshold 0.065** is the highest score of anyone else in the sample. Too low →
  a wrong person is confirmed (0.055–0.06 on the sample: 40–43 % right); too high → nothing is
  reported if the staff member's best tag score stays below it (on the sample anything from 0.065
  to 0.10 gives the same result, because one good confirmation is enough). If nothing is
  confirmed, the run says so in the log. An adaptive threshold is an open idea (the user's).
- No staff before the first confirmation, by design.
- Never tested on a video without a staff member, or from another camera.
