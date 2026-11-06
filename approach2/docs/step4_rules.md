# Step 4.1: rules for following the staff member

The state machine for 4.2–4.5, with every setting the user can tune. Built in 4.2
(`src/staffid2/follow.py`; results in `step4_follow.md`). The user said "go first, refine
later" (2026-10-26); the points marked **(4.2)** were added while building, see `step4_follow.md`. It is
**live**: frame *t* is decided using frames up to *t* only. The one exception is the saved CSV,
whose short gaps are filled at the end (decision 2).

Inputs per frame: the tracker's boxes (track ID, box) and each box's tag score (`score_tracks.py`
or `FrameScorer`; both use only that frame).

## Words

| Word | Meaning |
|---|---|
| **confirmation** | a box with score > `conf_thr` (0.065): the tag is clearly seen |
| **hit** | a box with score ≥ `hit_thr` (0.037): maybe a tag. Only counts **in context** (below) |
| **target** | a person we believe is staff. Several targets can exist at once |
| **continuation** | a track that plausibly carries on a lost target: starts soon after, close enough for walking speed, similar speed |

## States of one target

```
          confirmation on a new person
 (none) ─────────────────────────────────► CONFIRMED (green)
                                              │   ▲
                         its track ID ends    │   │ confirmation, or a hit on its own track
                                              ▼   │
                                            LOST (red cross at last position)
                                              │   │
           continuation found within          │   │ nothing within search_time
           search_time, but no tag yet        │   ▼
                                              │  DROPPED -> back to searching (if no other target)
                                              ▼
                                         UNCERTAIN (light green, dashed)
                                    (following a continuation; a hit on it -> CONFIRMED;
                                     its track ends -> LOST again)
```

- **CONFIRMED** and **UNCERTAIN** both mean "staff here" in the output. UNCERTAIN shows that we
  are following without having seen the tag since the last break.
- A target keeps following its **track ID** for as long as the tracker keeps it (tag hidden or not).
- **Focus** (orange) is colour only: a person with a hit now or in the last `focus_time`. It changes no decision.
- **Suspect** (yellow, 4.5 only): a box where a hit counts, after an overlap of the green target.
- A hit makes an UNCERTAIN target green again, but it is **not a confirmation**: only a score
  above `conf_thr` counts as a confirmation (and for choosing the main target). **(4.2)**
- An UNCERTAIN target has no time limit; it ends only when its track ends and nothing is found.
- A lost ID that comes back returns the target to the state it had before the loss.
- One track belongs to at most one target.

## What happens in each frame (in this order)

1. **Update targets.** For each CONFIRMED/UNCERTAIN target: if its track ID is in this frame,
   take the new box. If not, the target becomes LOST. Remember the last position, the speed and
   direction over the last `speed_frames`, and whether it was near the centre or standing still.
2. **Switch an UNCERTAIN target (4.2)** while its loss is recent (≤ `search_time`): if a newer
   track is now a better continuation of the same loss, follow that one instead. Live, the right
   track often appears a few frames after a wrong one, and a track's speed is known only after
   2+ boxes.
   **Search for lost targets.** For each LOST target, look at tracks not owned by another target.
   Take the most plausible continuation (rules below) → UNCERTAIN. If LOST for longer than
   `search_time` → DROPPED.
3. **Confirmations.** For every box above `conf_thr`:
   - its track belongs to a target → that target is CONFIRMED (refresh);
   - it would be a plausible continuation of a LOST target, or of a still-switchable UNCERTAIN
     target **(4.2)** → that target is found again (CONFIRMED on the new track);
   - otherwise → a **new target**. Other targets are never dropped because of it.
4. **Hits in context** (`hit_thr`): a hit on an UNCERTAIN target's own track → CONFIRMED.
   (In 4.5 also: a hit on a suspect after a split, or near where a target was lost.)
   Hits anywhere else only colour the box (focus).
5. **Output** this frame (below).

## Continuation rule (the generic search, 4.2)

Copied from `labeling.suggest`, which the user's labelling tool already used:
- the candidate track's first box comes at most `search_time` after the loss (or the lost ID comes
  back: ByteTrack can revive an old ID);
- distance from the last known position ≤ `base_dist` + `max_speed` × frames since the loss;
- best = smallest of: distance / allowed + `gap_weight` × gap / `search_time` + speed difference / `max_speed`.

**(4.2)** A candidate's speed is measured over its boxes seen so far; with one box it is not
counted yet. When an UNCERTAIN target's guessed track ends after fewer than `trust_frames`
(10) frames, candidates are still measured from the earlier loss point (the guess's position
is not trusted yet); a guess followed longer becomes the new loss point.

On its own this already covers part of the special cases: a person reappearing after the centre
blind spot, or after standing still, is usually a new track starting close by.

## Special cases (4.5, added only where 4.4 shows they are needed; green target only)

1. **Overlap / ID swap: suspects (built, see `step4_follow.md`).** When a target is lost, a
   suspect window opens (`suspect_time`). Everyone within `suspect_radius` of the loss point,
   **old tracks included**, becomes a suspect (yellow). The movement guess carries on as light
   green. A hit on a suspect that moved ≥ `suspect_move_px` over `move_frames` moves the target
   there; latest hit wins until a confirmation or the window ends. (The planned IoU / jump trigger
   did not fit the real swap: the nearest person was 129 px away with IoU 0.1.)
2. **Centre blind spot.** If the target is lost within `centre_radius` of the image centre, prefer
   a **newborn** track (an ID not seen before) appearing on the far side, in the direction of travel.
3. **Standing still / changing clothes.** If the target was slower than `still_speed` when lost:
   hold the last position for up to `hold_time`, and accept a box that appears within `still_dist`
   of it.

## Output

- **Per frame, per target:** state, x, y (box centre), track ID, reason
  ("confirmed: score 0.106", "continuation of 181: 0.4 s, 120 px", "lost at 354").
- **Main CSV** `frame,staff_present,x,y`: the **most recently confirmed** target (decision 1);
  `staff_present = 1` when it is CONFIRMED or UNCERTAIN. **(4.2)** It stays the main target while
  it is LOST: the live output then says 0 instead of jumping to an older target.
- **All-targets CSV:** one row per target per frame.
- **Filling at the end (decision 2):** after the whole video, a gap of at most `fill_max` frames
  *between two pieces of the same target* (lost, then found again) is filled. The fill is a
  straight line, or the held position when the target was standing still ("deduced"). Filled
  rows are marked `source = interp / held`. The live view is never filled.

## Colours (viewer, 4.3)

| Colour | Meaning |
|---|---|
| green, thick | CONFIRMED target |
| light green, dashed | UNCERTAIN target (following without a tag since the last break) |
| yellow | suspect (after a split, 4.5) |
| orange | focus: a hit now or in the last `focus_time` (colour only) |
| white, dotted | FILLED (saved view only) |
| grey | everyone else |
| red cross | last position of a LOST target while we search |

## Settings (all tunable, as flags / one config dataclass)

| Setting | Start value | Where it comes from |
|---|---|---|
| `conf_thr` | 0.065 | highest non-staff score on half of the sample's tracks (none of the other half above it). Main risk on a new video |
| `hit_thr` | 0.037 | 3c: 2 % of other people's crops pass |
| `search_time` | 62 frames (2.5 s) | same as the answer key's fill rule |
| `base_dist`, `max_speed`, `gap_weight`, `speed_frames` | 100 px, 30 px/frame, 3, 5 | `labeling.suggest`; staff walks up to ~28 px/frame |
| `fill_max` | 62 frames (2.5 s) | as `search_time` |
| `focus_time` | 25 frames (1 s) | colour only |
| `trust_frames` | 10 frames (0.4 s) | **(4.2)** a guessed track followed this long becomes the new loss point |
| `suspect_radius`, `suspect_time`, `suspect_move_px`, `move_frames` | 250 px, 125 frames (5 s), 40 px, 25 frames | 4.5 case 1; first guesses, not tuned |
| `centre_radius` | to measure | 4.5: from where tracks die and are born near the centre |
| `still_speed`, `still_dist`, `hold_time` | 2 px/frame, 40 px, 125 frames | 4.5; to be checked |

## Scoring (4.4, decision 3)

Against `ground_truth/ground_truth.csv`, "right" = within 25 px of the answer key's position:
1. **Time to first confirmation** (on the sample we expect frame 385).
2. **Correct when it says "staff":** of the frames with `staff_present = 1`, the share on the
   right person, and a list of wrong-person stretches.
3. **Coverage after the first confirmation:** of the frames after it, the share with the staff
   member found and right.

Reported separately for the live output (no filling) and for the saved CSV (with filling).

## Plan

- 4.2 `follow.py` with steps 1–5 and the generic continuation (no special cases), run frame by
  frame on the saved scores (the same as live, since every score uses only its own frame).
- 4.3 follow layer in the tag score viewer: the colours above + the reason text.
- 4.4 score it, look at the wrong stretches together in the viewer.
- 4.5 special cases one at a time, each re-scored.
- 4.6 robustness: higher `conf_thr`, strongest confirmations hidden (imitating a harder video).
