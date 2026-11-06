# Tag recognition & staff tracking: the rules in short

Plain summary of how the staff member is found and followed. Details: `step3c_ncc.md` (tag
score), `step4_rules.md` (full rules and settings), `step4_follow.md` (results). Code:
`src/staffid2/follow.py`.

## Tag score

Every tracked person box in every frame gets a tag score (NCC, step 3c): how much better the
tag's comb pattern fits the chest than a plain block of the same size. Each score uses only its
own frame. Two thresholds:

| Threshold | Value | Meaning |
|---|---|---|
| hit | ≥ 0.037 | maybe a tag; seated people also reach it now and then |
| confirmation | > 0.065 | the tag is clearly seen; no non-staff box in the sample reached it |

## Colours and states

| Colour | State | Meaning |
|---|---|---|
| orange | (hit) | a hit on this box now or in the last 1 s; colour only, changes nothing |
| green, thick | CONFIRMED | the tag was seen on this track |
| light green, dashed | UNCERTAIN | following our best guess of where the target went; no tag on it yet |
| red cross | LOST | the target's track ended; searching, at its last position |
| yellow | suspect | a person near where a target was lost; a hit on them moves the target there |
| grey | | everyone else |

## The rules

1. **Orange (hit).** Score ≥ 0.037 on any box. Colour only for 1 s; on a random person it
   changes nothing.

2. **Green (confirmation).** Score > 0.065 on any box:
   - it refreshes the target that owns that track, or
   - if it fits as the continuation of a lost or light-green target, that target is *found
     again* on this box, or
   - otherwise it becomes a **new target** (T1, T2, ...). Other targets are kept.

3. **Green stays green while the tracker keeps the same track ID**, even if the tag is not seen
   again for hundreds of frames. Only the track ID ending changes it.

4. **Losing the track gives a red cross, not light green.** When the track ID ends, the target is
   LOST: a red cross at its last position; the live output says "not present" for those frames.

5. **Picking the continuation.**
   - Candidates: the lost ID coming back (taken at once, the target gets its old colour back),
     or a **new track** that appeared after the loss, is not owned by another target, and is
     within 100 px + 30 px per frame since the loss.
   - Cost = how far (distance / allowed distance) + how soon (frames since the loss, weighted
     ×3, counts most) + speed similarity (once the new track has 2+ boxes). Lowest cost wins ->
     the target turns **light green** on it.
   - Nothing within 2.5 s -> the target is dropped and we are back to searching for the tag.

6. **Light green can change its mind.** For up to 2.5 s after the loss, if a better candidate
   appears, it switches.

7. **Light green -> green.** A hit (0.037) on its own track is enough, or a confirmation
   (0.065). A green made by a hit does not count as a confirmation when choosing the main
   target (rule 9).

8. **Light green's track can end too.** It goes back to a red cross and the search restarts.
   If that guessed track lasted under 10 frames, we search from the original loss point (the
   guess's position is not trusted); if 10+ frames, from where the guess ended.

9. **Output.**
   - Several targets can exist at once. The main CSV (`frame,staff_present,x,y`) reports the
     target with the **most recent real confirmation** (> 0.065). If that one is lost, the live
     output says 0; it does not jump to another target.
   - After the video, gaps of up to 2.5 s between two pieces of the same target are filled
     (straight line, or the held position if the person was standing still). The live output is
     never filled.

10. **Suspects (for ID swaps).** When a target is lost, everyone within 250 px of the loss point
    becomes a suspect (yellow) for 5 s, **including old tracks** (which the normal search never
    picks). The movement guess (light green) carries on meanwhile. A hit on a suspect that has
    been moving (≥ 40 px in the last second) moves the target there and turns it green. Latest hit
    wins: another suspect's later hit moves it again, until a real confirmation or the 5 s end.
    Normal search = a guess by movement, new tracks only; suspects = a shortlist where the tag
    decides, old tracks allowed. "Latest" means latest in time: a later hit wins even with a lower
    score (comparing scores would have kept the wrong person at the swap: 0.055 on 384 at 881 vs
    0.052 on the staff member at 890).
11. **While the suspect window is open, a target is shown light green** even after a hit (on its
    own track or on a suspect), because a later suspect hit can still move it. It turns solid green
    only on a real confirmation (> 0.065, which also closes the window) or when the 5 s end. On the
    sample every solid-green frame (196) is on the right person.

**In one line:** orange = a hint; green = the tag was seen on this track; light green = following
our best guess; red cross = lost and searching; yellow = nearby people the tag can pick. A hit only counts on a light-green target's own
track or a moving suspect.

## Known weak points (to refine)

- A single hit is enough to make a light-green guess green: at frame 644 it was the right person
  (track 270, score 0.044, during the clothes change), at 951 it was the wrong person (track 425,
  also 0.044). Of the 9 hits on light-green targets, 8 were right; right ones scored 0.040–0.064,
  the wrong one 0.044, so no hit threshold separates them. In the current rules a hit only changes
  the colour: hit_thr 0.045 or 0.05 gives exactly the same CSV.
- A target made green that way never expires (stale green box after the ID swap).
- ID swap at ~867: fixed for the most part by suspects (wrong live frames 132 -> 40), but another
  walking person near the staff member also gives hits, so the target is on them for ~10 frames
  twice before the staff member's next hit.
