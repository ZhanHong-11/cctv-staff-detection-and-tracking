# Step 4A: Detecting the tag (template matching + surround check)

**Code:** `src/staffid/tag.py`, `scripts/step4_make_templates.py`, `scripts/step4_score_tags.py`
**Run:** `python scripts/step4_make_templates.py && python scripts/step4_score_tags.py` (~1 min)
**Output:** `models/tag_templates/*.png` (committed), `data/step4/tag_scores.csv`
**Figure:** `docs/figures/step4_templates.png`

## The challenge
The tag is about **13×7 px** with bars 1–2 px wide, blurred by compression and motion
(`figures/step1_tag_pixels.jpg`). It appears at **any rotation** (fisheye) and on **light or dark
clothing**: the staff member changes from a white shirt into a black hoodie mid-video.

## 1. Rotate each person upright
`upright_crop()` rotates the person around their box centre so that the direction away from the
image centre points up. In an overhead fisheye, that's the direction from feet to head. The angle
comes from geometry alone (`atan2` of the box position relative to the image centre), so no learning
is needed. After rotation the tag bars always point the same way, so a few templates are enough. The
crop is also upscaled 2× to give the tiny tag a few more pixels to match against.

## 2. Templates from real frames
9 tag templates were cut from upright crops of the staff member: 2 on the white shirt, 7 on the
hoodie. Positions were marked by eye, then centred automatically on the nearest blob (bright for
the hoodie, dark bars for the shirt). See `figures/step4_templates.png`. **Assumption:** in the
interview the tag looks the same. If the design changed, only these 9 PNGs would need replacing.

## 3. Matching: normalised cross-correlation
`cv2.matchTemplate(TM_CCOEFF_NORMED)` at 3 scales × 3 small tilts (±12°) per template (81 variants),
searched only in the **chest window**: the area around the crop centre, sized from the person's
length (`chest_half_w`, `chest_above`, `chest_below`).

## 4. The surround check (key fix)
**First version without it: AUC 0.557, close to guessing.** Everyone scored high (median 0.86 for
other people against 0.88 for the staff member). Normalised correlation ignores contrast, so laptop
screen corners and monitor edges next to seated people looked as "tag-like" as the real tag
(`data/review/tag_top.jpg`).

The real tag differs in one way: it's a **small patch that contrasts with the clothing all around
it**. `surround_check()` looks at a ring around each candidate:
- **Dark clothing** (hoodie): the patch must be ≥ 35 grey levels brighter than the ring, and at most
  12% of the ring may be that bright. A laptop corner fails because it's the edge of a much larger
  bright area.
- **Light clothing** (shirt): the white tag blends in, so compare the **darkest 10% of the patch**
  (the bars) with the ring. The ring must be light (≥ 100) with at most 12% dark pixels.
  A first version compared averages and rejected every shirt tag, because thin bars barely lower the
  average (4–16 grey levels).

The top 10 candidates by correlation are checked in order. The first that passes gives the score;
if none passes, the score is 0. A too-large search window initially let one seated person's monitor
edge pass, which the person-sized chest window fixed.

## Results (boxes scored every 3rd frame, certain frames of the ground-truth draft)
| Version | Staff boxes with score ≥ 0.9 | Other boxes ≥ 0.9 | Notes |
|---|---|---|---|
| matching only | — | — | AUC 0.557 |
| + surround check | 17.5% | 0.93% | false hits concentrated on one seated person |
| + light-clothing rule + chest window | **17.9%** | **0.00%** | every hit is on a staff track |

**Only ~18% of staff boxes show the tag.** It's hidden when the person faces away, turns sideways,
or crosses the centre. That's expected, and it's why decisions are made per track (Step 5), not
per frame. The low hit rate isn't a weakness of the detector; the tag simply isn't visible in most
frames.

Speed: **13.5 ms per box**. The first version (5 scales × 5 tilts, 3× upscale, chest window not
limited) took 236 ms per box, which would have been ~50 min for the video.

## Honest caveats
- Templates **and** thresholds were chosen on this same video, so these numbers are optimistic. The
  robustness test (Step 8) and the unseen interview video are the real checks.
- Hand-made rules (35 grey levels, 12%) can break with different lighting, camera height or
  clothing colour (e.g. a grey shirt). A trained tag classifier (Step 4B: real tag patches + synthetic
  tags pasted onto other people's crops) would generalise better. That's the main next step.
