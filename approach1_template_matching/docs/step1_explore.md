# Step 1: Exploring the video

**Status:** exploration done. A ground-truth draft is built (see the end of this file) and
**needs checking by the user**.

## What was run
| Script | What it does | Output |
|---|---|---|
| `scripts/step1_explore.py` | Runs background subtraction (OpenCV MOG2) over the whole video. Saves every moving-blob box, and zoomed crops of the blobs every 10 frames. | `data/step1/motion_boxes.csv`, `data/step1/crops/` (not committed, can be regenerated) |
| `scripts/step1_figures.py` | Builds the figures below and measures the change between consecutive frames. | `docs/figures/step1_*` |

Logs are in `logs/step1_*.log` (not committed).

## Findings

### 1. The scene
960×720, 25 fps, 1341 frames (about 54 s), one overhead **fisheye** camera above an open office
(`figures/step1_contact_sheet.jpg`). About 20 people sit at desks. A few people walk along the
central corridor and the outer aisles.

**Why it matters:** in a fisheye image people look rotated depending on where they stand. Viewed
from the image centre, a standing person points outwards, like a clock hand. Anything that looks at
the tag (Step 4) has to cope with every rotation, or first rotate each crop upright using the angle
from the image centre.

### 2. The staff member appears in two outfits, and both carry the tag
(`figures/step1_tag_zoom.jpg`)
- **White shirt and black trousers**, roughly frames 330–640. The white tag blends into the white
  shirt, so only a row of **dark bars** is visible.
- **Black hoodie**, roughly frames 690–1200. The tag is a **white patch with dark bars**, the same
  as the first snapshot in the PDF.

These match the two snapshots in the PDF ("2 snapshot instances of the person"). The exact frame
ranges are estimates from browsing every 10th frame and still need checking.

**Why it matters:** clothing colour **can't** identify the staff member, because the same person
changes outfit. The detector has to look for the bar pattern itself, and it has to work on both
light and dark clothing. Looking for "a small white rectangle" alone would miss the white-shirt
part.

### 3. The tag is tiny
(`figures/step1_tag_pixels.jpg`, 10× zoom with a 5 px grid)
- On the hoodie: about **13×7 px**, with bars 1–2 px wide.
- On the shirt: about **12×4 px** of dark marks.
- It's also blurred by video compression and motion.

**Why it matters:** a single frame often can't settle the question, and many frames show no tag at
all (the person is facing away, an arm covers it, or they're at the fisheye edge). The design
therefore **follows each person over time and combines the evidence** (Step 5), instead of
expecting the tag in every frame.

### 4. People who stand still disappear from background subtraction
Background subtraction finds walkers well. But around frames 660–810 a dark-clothed person stands
almost still near the bottom desks and gets no box. At frame 950 the box also landed on the wrong
area. Two walkers that meet can merge into one blob (frames 900–910).

**Why it matters:** background subtraction is a usable fallback that needs no installs, but a
proper **person detector (YOLO)** is needed so that someone standing still is still counted as
present.

### 5. Exposure dip at frames 1300–1305
(`figures/step1_frame_diff.png`, `figures/step1_exposure_dip.jpg`) The average difference between
consecutive frames is normally about 1.4, but it jumps to 99.6 at frame 1300. The picture goes
almost black for a few frames, then recovers. It's the **same camera and scene**, most likely
auto-exposure or a dropped segment.

**Why it matters:** a background model has to be reset after a jump like this, otherwise it
reports the whole frame as foreground. Unseen test videos may contain similar glitches, so the
final pipeline should detect large global brightness changes and handle them.

## Assumptions so far
- Anyone carrying the tag is staff. The pipeline won't assume there is only one staff member, so it
  also works on a test video with zero or several.
- "Staff present" means at least part of the staff member is visible in the frame.

## Ground truth draft (drafted by Claude, needs checking by the user)
Labeling 1341 frames box by box by hand is slow, so the draft reuses the Step 2–3 tracks:
1. Contact sheets of the moving tracks (`scripts/review_tracks.py`) were checked by eye to decide
   which tracks show the staff member, with the tag visible in many of them.
2. The decisions are in **`approach1_groundtruth/staff_chain.json`**: which track follows the staff member
   over which frames, and why. Edit this file to correct the ground truth.
3. `scripts/step1_build_ground_truth.py` turns it into **`approach1_groundtruth/ground_truth_draft.csv`**:
   `frame, staff_present, x, y, x1, y1, x2, y2, source, certain, pos_certain`.
   Frames between pieces are interpolated, since the staff member was seen crossing the image
   centre in every gap checked (`data/review/gaps.jpg`). It also writes
   `data/review/ground_truth_draft.mp4` for checking.

### What the draft says
| Frames | Staff | Certain? | Notes |
|---|---|---|---|
| 0–274 | present, **seated** at the far-left desk in a white shirt | **no** | tag not visible; identity inferred because the staff member gets up from this seat |
| 275–1257 | present, walking (white shirt, then hoodie from ~640) | yes | puts the hoodie on at ~633–653 while standing at the bottom desks |
| 1258–1340 | absent | **no** | walked out at the bottom-left; could be sitting out of sight |

1258 frames present, 983 with certain presence, 885 with certain position (`pos_certain` = from a
track or a gap ≤ 10 frames; long gaps are guesses, e.g. frame 575 lands on empty floor).

### Things to check (user)
1. Is the seated white-shirt person in frames 0–274 really the staff member?
2. Does the staff member really leave at ~1257, or sit down somewhere visible?
3. Spot-check `data/review/ground_truth_draft.mp4` (green = tracked, yellow = interpolated).

### Another finding from making the ground truth
There's a **second dark-clothed person (not staff)** who walks near the staff member around frames
900–1210. At ~1203 they meet, and the tracker's ID for the staff member jumps onto this person. This
is a realistic hard case: clothing alone can't tell them apart, only the tag can.
