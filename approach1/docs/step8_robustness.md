# Step 8: CLI and robustness check

## The command
`python detect_staff.py --video <file>` runs everything in one read of the video (detect → track
→ tag score every 3rd frame), then links, votes and writes `outputs/<name>/`. See `README.md`.
On the sample video: **145 s analysis** for 54 s of video (M-series Mac, GPU via MPS). `--fast`
uses yolo11n and `--no-video` skips the annotated video.

Generic choices for unseen videos:
- The fisheye centre defaults to the image centre, and frame size and fps are read from the file.
- No frame numbers, regions or colours from the sample video are used in the pipeline.
- The tag templates are the only video-specific input (they describe the tag, not the scene).

## Simulated unseen videos
There's only one real video, so `scripts/step8_robustness.py` makes altered copies and runs the
unchanged pipeline on each: **mirrored, rotated 90° (portrait 720×960), shrunk to 75%, darkened to
55%**. Predicted positions are mapped back to original coordinates and scored on the certain
frames of the ground-truth draft.

**Limitation:** the people, clothes, tag and room are the same, so this only tests geometry, scale
and lighting. It can't show how the system does with new people or a new camera height.

## Finding: a single bad link can spread the staff label to the wrong person
First results (default voting):

| Variant | Recall | Position within 25 px | Staff chains |
|---|---|---|---|
| original | 0.897 | 94.9% | 2 |
| mirrored | 0.643 | 97.1% | 1 |
| rotated 90° | 0.997 | **61.0%** | 3 |
| shrunk 75% | 0.647 | 98.0% | 1 |
| darkened | 0.977 | **54.0%** | 3 |

The high recall for rotated/darkened was **misleading**: the staff member is present almost all the
time, so any chain labelled staff "hits" those frames. Only the position metric reveals that the
box sat on someone else. A per-chain check showed what happened:
- The true staff chains were 90–100% on the staff member in every variant.
- Each bad variant had **one long chain (frames ~156–1250) that was on the staff member only 8–11%
  of the time**. Its 4–8 tag hits were real (the staff member at ~1160–1203), but its tracks had
  been joined, through near links and a tracker ID swap, to a seated or other person covering ~1000
  frames.

## Fix: a minimum hit rate per chain
Tested on cached tracks/scores (`scripts/step8_revote.py`, seconds per rule):

| Rule (recall / within 25 px) | original | darkened | rotated 90° | mirrored | shrunk |
|---|---|---|---|---|---|
| baseline | 0.90 / 95% | 0.98 / 54% | 1.00 / 61% | 0.64 / 97% | 0.65 / 98% |
| no near links | 0.90 / 95% | 0.67 / 82% | 0.81 / 96% | 0.64 / 97% | 0.65 / 98% |
| hit window ±250 frames | 0.90 / 95% | 0.64 / 84% | 0.81 / 91% | 0.64 / 97% | 0.65 / 98% |
| **hit rate ≥ 3% (chosen)** | **0.90 / 95%** | **0.56 / 93%** | **0.67 / 97%** | **0.64 / 97%** | **0.65 / 98%** |
| no near links + hit rate ≥ 3% | 0.90 / 95% | 0.32 / 64% | 0.81 / 96% | 0.64 / 97% | 0.65 / 98% |

**Chosen: a chain needs tag hits on ≥ 3% of its scored boxes** (and ≥ 2 hits). The real staff
member shows the tag regularly; a long chain with only a handful of hits most likely contains an ID
swap. None of the rules changed the original video's result, so they don't hurt the case they
weren't tuned on.

**Why favour placing the label correctly over recall:** for footfall counting, calling a customer
"staff" removes a real visitor from the counts, and the x,y output would follow the wrong person.
A missed stretch only means some staff frames are counted as customers. Both matter, but a wrong
identity is worse and harder to notice.

**Caveat:** 3% was chosen on variants of the same video. It's still tuning on one clip.

## Remaining weaknesses seen here
- **Mirrored / shrunk: recall 0.64–0.65.** The white-shirt part (frames ~365–560) gets too few tag
  hits, so that chain isn't confirmed. The light-clothing tag is the weakest case: thin dark bars,
  and at 75% scale only ~9 px wide.
- **The darkened video** loses tag contrast: the fixed 35-grey-level rule is weaker in dim light.
  Normalising contrast per crop, or a trained classifier, would help.
- **No test with zero staff present.** The system has never been checked on a video where nobody
  wears the tag. A no-staff clip would be the first thing to test with more data.

## What would most improve robustness (next steps)
1. **A trained tag classifier (4B)** on real tag patches plus synthetic tags pasted onto other
   people's crops at random rotation, scale and brightness. It would replace the hand-set contrast
   rules.
2. **Appearance-based re-identification** in tracking, to stop ID swaps at the source.
3. **More videos**: the company's few hundred samples, to tune thresholds across scenes rather than
   on one clip.
