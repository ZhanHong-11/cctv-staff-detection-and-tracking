# Step 1: Exploring the video

**Status:** exploration done. A ground-truth draft is **not started yet**. It will be built once a
person tracker exists (see "Plan for ground truth" below).

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

## Plan for ground truth (next)
Labeling 1341 frames box by box by hand is slow. The plan:
1. Run the person tracker (Steps 2 and 3) to get tracks.
2. Make contact sheets per track and decide by eye which tracks are the staff member, then fix ID
   switches and gaps.
3. Write `data/ground_truth.csv` (`frame, staff_present, x, y, w, h`), marked as **drafted by
   Claude**.
4. **The user checks it** before it's used in the Step 7 metrics. Otherwise the same system would
   be writing the answer key and being graded against it.
