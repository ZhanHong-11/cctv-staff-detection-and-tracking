# FootfallCam AI Evaluation: Staff Identification Plan

> **Status (2026-10-24):** this is the original plan, written before any work. Steps 1–8 are
> done; 4B (trained tag classifier) wasn't built, and step 9 (the write-up) is the user's. For
> what was actually built and how it differs from this plan, see `README.md`, `JOURNEY.md` and
> `docs/`.

## What the task asks

**From the PDF:** a person counts as staff if they wear a small staff name tag. The tag is a white rectangle with black vertical bars, a bit like the letter "Ш", worn on the chest. You have to:

1. Find which frames contain the staff member. This is the main task.
2. Give the staff member's x,y coordinates in each of those frames. This is the bonus.
3. Write 1–2 pages of documentation with visuals and submit your code.
4. In the interview, run your code live on your laptop on a **new test video**. Plan around this one, because your solution has to work on a video you haven't seen.

**The video (`AI Test Candidates.mp4`):**

| Property | Value | What it means for you |
|---|---|---|
| Resolution | 960×720 | The tag is only about 8–20 px wide, so finding it is a small-object problem |
| FPS / length | 25 fps, 1341 frames (about 54 s) | Small enough to label by hand and check every frame |
| View | Overhead **fisheye** camera above an office | People look stretched near the edges, and normal person detectors (trained on side views) do worse |
| Scene | Many people sit still at desks; only a few walk the central corridor | Movement is a strong signal, since the staff member "walks along the corridor" |
| Walkers seen | A white-shirt person around frames 390–620, and dark-clothed people around frames 840–1180 | Both outfits match the two snapshots in the PDF (black top, white long-sleeve shirt) |

The tag could not be confirmed by eye at this resolution, so which walker is the staff member is still open. Step 1 settles that.

**The key insight:** the tag is tiny and often hidden, since the person turns, swings an arm or walks under the fisheye edge. Looking for the tag frame by frame will miss a lot. A better approach is to **follow each person across frames and decide "staff or not" for the whole track**. The tag only needs to be seen clearly in a few frames, and the label then carries over to every frame where that person appears.

---

## Step-by-step plan

### Step 1: Explore the video and build ground truth (about 2–3 h)

- **How:** Extract every frame. Find the frames where the tag is actually visible by zooming in on the corridor walkers. Then label the staff member's bounding box in every frame where they appear, using CVAT or Label Studio (both free), or a quick OpenCV click tool. Also crop 20–50 close-ups of the tag.
- **Why:** You can't measure accuracy or tune thresholds without ground truth. The tag crops also become your template and training data. Labeling 1341 frames is manageable: you only need to label the frames with a walker, and you can draw boxes on a few keyframes and interpolate between them.

### Step 2: Detect people (about 2 h)

- **How:** Start with a pretrained person detector such as YOLO11/YOLOv8 from `ultralytics`. Check how many people it finds and misses, especially at the fisheye edges. If recall is poor, try:
  - (a) a model trained on overhead or fisheye data (search for the "CEPDOF" / "WEPDTOF" fisheye people datasets),
  - (b) cutting the image into overlapping tiles and detecting on each (SAHI), or
  - (c) a fallback that treats moving blobs from background subtraction (`cv2.createBackgroundSubtractorMOG2`) as people.
- **Why:** Every later step works on person crops, so a missed person here means a missed staff frame. Background subtraction suits this scene because the seated people are static and the corridor walkers move.

### Step 3: Track people across frames (about 1–2 h)

- **How:** Use ByteTrack or BoT-SORT, which come built in with `ultralytics` (`model.track(..., tracker="bytetrack.yaml")`). Each person gets a track ID that stays the same across frames.
- **Why:** This is what makes track-level voting possible (Step 5). It also smooths out frames where detection misses someone for a moment. Watch for one person splitting into two IDs (ID switches) when they pass behind desks. Raising the tracker's `track_buffer` helps, and so does merging tracks that end and start close together in time and space.

### Step 4: Detect the tag inside each person crop (about 4–6 h, the core of the work)

Try two options and compare them on your Step 1 labels.

- **A. Hand-coded detector, as a baseline.** In each person crop, find small bright rectangles with OpenCV (thresholding, then `findContours`), keep only those with the right size and width-to-height ratio, then check for the dark-bar pattern inside them, either by counting bars along a line through the box or by template matching against your tag crops. It's quick to build, easy to explain, and needs no training data. It will produce false hits on white paper, screens and white shirts, but the next step filters many of those out.
- **B. A small trained model.** Either a classifier that labels each person crop "tag / no tag" (a small CNN such as MobileNet or ResNet18), or a YOLO model trained to detect the tag itself, run on upscaled crops. You'll have little data, so:
  1. apply heavy augmentation: rotation (tags appear at every angle in a fisheye image), blur, brightness changes and scaling, and
  2. **create synthetic training data** by pasting the tag onto person crops from people who aren't staff. This "copy-paste" trick gives you hundreds of positive examples from a few real ones, and it's the kind of creative idea the PDF is asking for.
- **Why:** The tag is the only feature that defines "staff". Clothing is not reliable, because the two snapshots show the person in different outfits. Comparing the two options also gives you good material for the write-up.

### Step 5: Vote at track level and label each frame (about 1–2 h)

- **How:** For each track ID, combine the tag scores from every frame, for example "at least N confident tag hits" or "average of the top-k scores above a threshold". Mark the winning track as staff, and mark **every frame** of that track as "staff present". Smooth the result over time: fill gaps of a few frames and drop single-frame blips.
- **Why:** This changes the problem from "see the tag in every frame", which can't work because the tag is often hidden, into "see it a few times per person", which can. It's the biggest single accuracy gain, and the main point to stress in your documentation.

### Step 6: Output x,y coordinates, for the bonus (about 1 h)

- **How:** For each staff frame, output the center of the bounding box as pixel coordinates. Overhead, the center of the box is roughly above the person's position on the floor. Optionally, describe or implement fisheye undistortion (`cv2.fisheye.undistortPoints`) to convert pixel positions to floor-plane coordinates. Without the camera's calibration you'll have to state your assumptions.
- **Why:** Pixel x,y is enough to answer the question. Mentioning floor-plane conversion shows you understand how FootfallCam actually uses this data (paths, dwell time, heatmaps).

### Step 7: Evaluate (about 1 h)

- **How:** Compare against your Step 1 labels:
  - **Frame-level** precision, recall and F1 for "staff present", plus a timeline chart of predicted against true intervals.
  - **Coordinate error:** average pixel distance from the true center, or the share of frames where the predicted box overlaps the true box by at least half (IoU ≥ 0.5).
  - Note which failure cases remain: tag hidden, ID switches, false hits from white objects.
- **Why:** Numbers make your claims believable, and discussing failure cases honestly comes across well in interviews.

### Step 8: Make it robust for the live test video (about 2 h, don't skip this)

- **How:** Wrap everything in one command-line script:

  ```
  python detect_staff.py --video test.mp4 --out results/
  ```

  It should produce a CSV (`frame, staff_present, x, y`), an annotated MP4 with the staff box highlighted, and a timeline PNG.
  - Don't hard-code any regions, frame numbers or colors from this video.
  - Keep it fast on a laptop CPU. Use a nano-size model, process every 2nd frame if you need to and fill in the frames between, and show a progress bar.
  - Pin your dependencies in `requirements.txt` and do a test run in a fresh venv.
  - Test it on a modified version of this video (flipped, cropped, re-encoded) to simulate an unseen one.
- **Why:** The interview is a live demo on new data. A solution that only works on this one video will fail in front of the interviewers. A solution that is a little less accurate but works reliably is better.

### Step 9: Write the documentation (about 2 h)

- **How:** 1–2 pages covering:
  1. The problem and your assumptions (one staff member, tag defines staff, fisheye view, approximate coordinates).
  2. A pipeline diagram: detect people → track → detect tag → vote per track → output.
  3. **Key parts to highlight:** track-level voting and the synthetic tag-pasting augmentation.
  4. Results: the metrics, the timeline chart and 2–3 annotated example frames.
  5. Limitations and next steps: training on your "few hundred videos", fisheye-specific detectors, using the sensor's depth data, and re-identification across cameras.
- **Why:** Reviewers skim, so the diagram and the visuals do most of the work.

---

## Suggested order and time budget

**Steps 1 → 2 → 3 → 4A → 5 → 7 → 8** give you a complete working baseline (roughly 1.5–2 days). Then add **4B and 6** to improve it, and finish with **9**. Get the end-to-end pipeline working first and improve it after. A finished simple pipeline beats an unfinished clever one.

## Tools

- Python, `opencv-python`, `numpy`, `pandas`, `matplotlib`
- `ultralytics` (YOLO + ByteTrack)
- Optional: `sahi` (tiled detection), `torch` / `torchvision` (tag classifier)
