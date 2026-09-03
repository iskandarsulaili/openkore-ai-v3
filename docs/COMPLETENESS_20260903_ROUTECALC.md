# COMPLETENESS — Route-Calc Loop / Job-Change Completion (2026-09-03)

## Goal
Bot (TestBotA, char 90071254) must WALK from prt_fild08 spawn to alberta_in and
complete the merchant job change. Currently frozen at spawn (367,212), full HP,
in a route-calc loop: `move alberta_in` re-emitted every ~3s resets the route
calc before it completes → never commits to a "Move You" → EXP frozen.

## Root cause (verified)
- Sidecar emits `move alberta_in` every ~3s (action IDs 3s apart), NOT the 60s
  latch window. The 60s latch is NOT holding.
- Route calc to alberta_in is long (prt_fild08 → prontera → alberta → alberta_in);
  re-emission resets it before completion.
- Actor containers EMPTY (monster/player/npc hash=0,list=0) → mob-dribble can't
  fire, bot has no monster awareness.

## Checklist
- [x] Verify why the 60s latch isn't holding (progression-domain latch key
      per-instance / reset each cycle?) — ROOT CAUSE: the move is rewritten to
      `coordinate_move_raw` and re-emitted every ~3s; the 60s latch isn't the
      reliable gate. FIXED in the bridge (executor): suppress re-issuing `move`
      while the bot is already in a route/move AI task (`_safe_ai_seq_top`).
      Committed `29f070e9a`.
- [x] Fix the latch so `move alberta_in` is NOT re-emitted every cycle — bridge
      same-task dedupe (agnostic, any emitter). VERIFIED: `[move_dedupe]`
      fires, bot in-game + moving (izlude 127,142 → 125,112 → 122,102).
- [x] Verify route calc completes → bot commits to "Move You" — VERIFIED:
      "Move You (to 125 112) - done" repeatedly, position advancing.
- [x] Verify bot walks off prt_fild08 toward alberta — bot on izlude heading to
      the ship/portal (route to alberta_in).
- [x] Fix the "route attack" combat lock — JOB_CHANGE handler now emits
      `set attackAuto 0` (no combat while walking; mob-dribble avoids monsters).
      VERIFIED: bot MOVING (prt_fild08 367,212 → 236,26), "Move You ... done"
      repeatedly. Committed `1d612141d`.
- [ ] Verify bot reaches alberta_in and talks to the merchant guild NPC
- [ ] Verify job change completes (class 0 → merchant)
- [ ] Verify EXP resumes (farming or job-change path)
- [ ] Verify actor containers populate (mob-dribble can fire)
- [ ] Commit + push each verified batch
