# Steps 5–7: Linking, voting, coordinates and evaluation

**Code:** `src/staffid/vote.py`, `src/staffid/evaluation.py`, `scripts/step5_vote.py`
**Run:** `python scripts/step5_vote.py` (seconds), ablations: `--no-link`, `--no-fill`
**Output:** `data/step5/chains.csv`, `staff_frames.csv`, `metrics.json`

## Why vote per track
The tag shows in only ~18% of the staff member's frames (Step 4), but where it does show, nobody else
produces hits. So the question changes from "is the tag in this frame?" to **"did this person show
the tag at some point?"**. Once a person is identified, all their frames count as staff.

## 1. Linking track pieces (`link_tracks`)
The staff member's track breaks into ~20 pieces, mostly where they cross the image centre (Steps
2–3). Voting per piece would miss every piece where the tag happened to be hidden.

**Calibration:** the true hand-offs from the ground truth have gaps of **7–14 frames** but jumps of
**150–390 px** (21–33 px per frame). In fisheye pixels the centre crossing is fast. A plain distance
limit wide enough for this links almost anything. The first version did, and the median position
error was 250 px because chains mixed several people.

**Rule used:** two kinds of links, resolved greedily, best first, each track end and start used once:
| Link | Condition | Covers |
|---|---|---|
| near | ≤ 60 px, gap ≤ 40 frames | a person who stayed in place (track broke) |
| jump | gap ≤ 25 frames, ≤ 35 px/frame, **and** the jump direction matches both the end motion of the first piece and the start motion of the next (cos ≥ 0.8, ~37°) | walking through a blind spot |

The direction test is the key: someone walking through the centre keeps their heading, while
seated people and random new tracks don't. Result: 159 tracks → 107 chains, and the staff member's
pieces end up in 2 chains (142→…→329 and 333→…→385). That's almost exactly the hand-made ground
truth.

## 2. Voting (`vote`)
A chain is staff if it has **≥ 2 hits** (tag score ≥ 0.9). Two hits protect against a single
chance match. On this video every non-staff chain has 0 or 1 hits.

## 3. Per-frame output and x,y (`staff_frames`) — Step 6
- Staff present = any frame covered by a staff chain, including the **gaps inside the chain**. The
  person can't leave the room through the middle of the image, so they're in view while crossing
  the centre.
- **x,y = centre of the person's box** in pixels (origin top-left). In gaps, the box is linearly
  interpolated. Overhead, the box centre is roughly above the person's position on the floor.
- If two staff chains overlap in a frame, the one with more hits is reported.

**Not done: floor coordinates.** Converting pixels to metres on the floor needs the fisheye lens
calibration (or a few known floor points), which we don't have. With it,
`cv2.fisheye.undistortPoints` plus a floor homography would give metres.

## 4. Evaluation (Step 7) against the ground-truth draft
"Certain frames" excludes frames where the staff member's identity can't be confirmed (seated
0–274, after 1257). See `docs/step1_explore.md`.

| Variant | Recall | Precision | Position within 25 px |
|---|---|---|---|
| vote per track only | 0.626 | 1.0 | 97.9% |
| + fill gaps (per track) | 0.714 | 1.0 | 93.8% |
| + linking (no fill) | 0.688 | 1.0 | 98.1% |
| **+ linking + fill (final)** | **0.897** | **1.0** | **94.9%** (median 0 px) |

**Predicted:** staff present in frames **365–1246** (14.6 s – 49.8 s), 882 frames.

### What's still wrong and why
| Frames | Error | Cause |
|---|---|---|
| 275–364 | missed (white shirt, walking off) | the link 122→142 didn't form, and those pieces show no tag (facing away) |
| 1209–1257 | missed / wrong box at ~1211–1246 | track 385 swapped to another dark-clothed person, so the true continuation (414) can't be linked by direction |
| 0–274 | not detected (excluded from metrics) | seated, tag never visible: undecidable from the video |

### Caveats about the metrics
- The ground truth is **drafted by Claude** and needs checking by the user.
- The staff member is present in almost every frame, so there are no certain "absent" frames
  (TN = 0). Precision here means "never placed the staff label on someone else", which the
  position metric confirms. It **doesn't** show how the system behaves on a video with no staff
  member. The robustness test (Step 8) and the interview video are needed for that.
- Parameters were tuned on this video, so expect lower numbers on unseen data.
