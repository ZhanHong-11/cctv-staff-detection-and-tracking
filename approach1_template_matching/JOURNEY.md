# How the solution evolved: what was tried, what broke, what changed

A summary of the whole process, from the first look at the video to the final pipeline. Each stage
follows the same pattern: **what I tried → the problem → what that taught me → how the strategy
changed.** The day-by-day record is in `WORKLOG.md`, the per-step details are in `docs/`, and every
run left a log in `logs/` (local only).

---

## 0. Starting point: the plan
**Plan:** detect people → track them → find the tag on each person → decide per person → output frames
and x,y. Key idea from the start: the tag is tiny and often hidden, so **decide per person
(track), not per frame**.

---

## 1. Understanding the video
**Tried:** contact sheets of frames, background subtraction (OpenCV MOG2) to find moving people,
zooming into crops at 10×.

**Found:**
- The camera is an **overhead fisheye**: people look rotated depending on where they stand.
- The staff member appears in **two outfits**: a white shirt, then a black hoodie. Both carry the tag.
  → *Clothing can't identify the staff member; only the tag can.*
- The tag is about **13×7 px**, bars 1–2 px wide. → *Often unreadable in a single frame.*
- Background subtraction **misses people who stand still** and merges people walking close together.
  → *Need a proper person detector.*
- An **exposure dip** at frames 1300–1305 (the image goes almost black). → *The pipeline must cope
  with glitches.*

**Strategy change:** drop the motion detector as the main approach and use YOLO. *(Blocked briefly:
Claude Code's safety check refused the package install. Solved by putting everything in
`requirements.txt` for you to install once.)*

---

## 2. Detecting people: YOLO failed on half the image
**Tried:** YOLO11 (nano and small) on the raw frames.

**Problem:** it found only **0–5 of ~20 people** per frame, all in the top part of the image. It
missed the staff member completely at frames 750 and 950.

**Insight:** YOLO learned from normal photos of upright people. In an overhead fisheye, people at the
bottom of the image look **upside down** and people at the sides lie sideways.

**Change:** **rotate each frame 0/90/180/270°**, detect on all four, rotate the boxes back and merge
duplicates. Everyone is upright in at least one rotation. → **11–12 people per frame**, at almost no
extra time (the 4 images run as one GPU batch).

**Remaining blind spot:** directly under the camera (image centre) people are seen from straight
above, just a head and shoulders, so YOLO still misses them there.

---

## 3. Tracking + building an answer key
**Tried:** ByteTrack to follow each person across frames → 159 tracks.

**Found while checking tracks by eye:**
- The staff member's track **breaks into ~20 pieces**, every time they cross the image centre.
- The staff member **puts the hoodie on at frames ~633–653** while standing still. One person, not two.
- There's a **second dark-clothed person (not staff)**, and at frame ~1211 the tracker's ID **jumps from
  the staff member onto them**.
- The seated white-shirt person in frames 0–274 is *probably* the staff member, but the tag can't be
  seen. → marked **uncertain**.

**Change:** built the answer key (`ground_truth/`) from the tracks with my decisions written down
and explained, so you can check them. Positions in long gaps turned out to be guesses (frame 575
landed on empty floor), so I added a `pos_certain` column.

---

## 4. Recognising the tag: four attempts
### Attempt 1: template matching. Too slow.
**Tried:** rotate each person upright (the angle comes from fisheye geometry), then slide 9 real tag
templates over the chest at 5 sizes × 5 tilts.
**Problem:** **~50 minutes** for one video (236 ms per person box). You asked whether it was stuck;
it wasn't, just slow.
**Change:** 3 sizes × 3 tilts, smaller upscale, score every 3rd frame → 33 ms per box, ~2.5 min.

### Attempt 2: the faster version. No better than guessing.
**Problem:** **AUC 0.557**, close to a coin flip. Everyone scored high (median 0.86 against 0.88 for
the staff member).
**Investigated:** the top matches on non-staff people were **laptop screen corners and monitor
edges**. The matching method ignores contrast, so any small light-dark pattern looks "tag-like".
**Insight:** the real tag is a **small patch that contrasts with the clothing all around it**; a
laptop corner is the edge of a larger bright area.
**Change:** added a **surround check** on a ring around each candidate.

### Attempt 3: surround check. Better, but the white shirt failed.
**Problem:** other people's false hits dropped sharply, but **the white-shirt tag never passed**.
**Investigated:** on the white shirt the match was excellent (0.95–0.99), but thin dark bars barely
change the *average* brightness of the patch (only 4–16 grey levels).
**Change:** for light clothing, compare the **darkest pixels** (the bars) with the ring instead of
the average.

### Attempt 4: plus a person-sized search window. Works.
**Problem left:** one seated person kept matching a monitor edge **above their head**, outside their body.
**Change:** size the search window from the person's box, so only the chest is searched.
**Result:** **0 false hits on other people; ~18% of the staff member's boxes show a hit.** The tag
is simply hidden the rest of the time (facing away, turning, crossing the centre).

---

## 5. Deciding who is staff: linking track pieces
### Attempt 1: link pieces by distance. Mixed people together.
**Tried:** join a track that ends to one that starts soon after and nearby, then vote (≥ 2 hits = staff).
**Problem:** presence looked great, but the **median position error was 250 px**: the box was often
on someone else. The distance limit was so loose that chains joined several people.

### Attempt 2: calibrate on real hand-offs, then add direction
**Investigated:** the true staff hand-offs in the answer key: gaps of **7–14 frames**, but jumps of
**150–390 px**. Crossing the fisheye centre is fast in pixels, so distance alone can't tell right
links from wrong ones.
**Insight:** someone walking through the centre **keeps their direction**. A random new track
doesn't line up with where the first track was heading.
**Change:** a "jump" link must continue the end direction of the first piece and match the start
direction of the next.
**Result:** 159 tracks → the staff member ends up in **2 clean chains**. Precision 1.0, recall 0.90,
95% of positions within 25 px. An ablation showed linking + gap filling add ~27 recall points.

---

## 6. Packaging: one command
**Built:** `detect_staff.py`, a single read of the video that writes the CSV, timeline, annotated
video and summary. 2.5 min for the 54 s video. Later switched the video to H.264 so it plays in
QuickTime (and it went from 40 MB to 10 MB).

---

## 7. Robustness test: found a hidden bug
**Why:** the interview uses an **unseen video**, and every setting so far was tuned on one clip.
**Tried:** ran the unchanged pipeline on altered copies: mirrored, rotated 90°, shrunk to 75%, darkened.

**Problem:** rotated and darkened looked *perfect* on presence (recall ~1.0) but had **only 54–61%
of positions correct**.
**Insight 1, about the metric:** the staff member is in view almost all the time, so *any* chain
labelled staff scores well on presence. **Only the position check reveals a wrong identity.**
**Insight 2, the cause:** one long chain (~1000 frames) showed the tag only 4–8 times. It was the
staff member only briefly, joined through a tracker ID swap to someone else. One bad link spread
the staff label across the video.

**Tried 5 fixes on cached results** (seconds per test instead of 10 minutes): no near links, a time
window around hits, a minimum hit rate, and combinations.
**Change:** **a chain must show the tag in at least 3% of its checked frames.** The real staff
member shows it regularly; a long chain with only a few hits contains a swap. → **93–98% of
positions correct on every variant**, with no change on the original video.
**Trade-off accepted:** lower recall on the hard variants (0.56–0.67). Calling a customer "staff" is
worse than missing a stretch: it corrupts both the footfall count and the x,y output.

---

## Final state
| | Original video (certain frames) |
|---|---|
| Staff present | frames 365–1246 (14.6–49.8 s) |
| Precision / recall | 1.0 / 0.90 |
| Position within 25 px | 95% |
| Runtime | ~2.5 min for 54 s of video |

**Known gaps:** the white-shirt walk-off (275–364) is missed; the ID swap at ~1211 puts the box on
the wrong person until ~1246; the seated period (0–274) can't be decided from the video; there's
no test on a video without staff; everything is tuned on one clip.

## The thread through all of it
1. **Look before fixing.** Every real improvement came from inspecting the failures (upside-down
   people, laptop corners, thin bars, the ID swap), not from adjusting numbers blindly.
2. **Work around the hidden-tag problem instead of fighting it.** The tag shows in only ~18% of
   frames, so identify the *person* once and carry the label along their track.
3. **Distrust metrics that look too good.** Presence scores hid a wrong identity; only the position
   check exposed it.
4. **Prefer simple rules you can explain**: rotations, a surround check, direction-consistent
   links, a minimum hit rate. Each one fixes a specific failure you can point to.

## What I'd do next
1. Train a small tag classifier on real tag patches plus synthetic ones pasted onto other people.
   It would replace the hand-set contrast rules, the most fragile part.
2. Add appearance-based re-identification to stop ID swaps at the source.
3. Tune and test on more videos, including one with no staff member.
