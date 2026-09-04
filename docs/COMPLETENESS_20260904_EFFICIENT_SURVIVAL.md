# COMPLETENESS — Efficient Survival Strategy (2026-09-04)

## Goal
Fix the LLM survival advisory prompt so it encodes the efficiency insight:
job-changing ASAP is the priority (a new job's job-level farming is strictly more
efficient than novice base-level farming). `level_up_first` must be ONLY a brief
zeny-farm to afford the escape (Fly Wing), never a long novice grind.

## Root cause (user-corrected)
- LLM decided `level_up_first` (farm safe map) when the bot was broke (0 zeny,
  0 Fly Wing) and couldn't cross the field to job change.
- But delaying job change is INEFFICIENT: a lv31 novice's base-EXP gains don't
  build the new job's job level. Job-changing NOW means every kill builds the
  merchant's job level.
- The LLM prompt framed it as "level up first vs job change now" — missing that
  job-change ASAP is always better; the only question is how to afford the escape.

## Checklist
- [x] Update `_llm_survival_advisory` prompt: job-change ASAP is the priority;
      `level_up_first` = brief zeny-farm to afford the escape, never a long grind.
      VERIFIED LIVE: LLM decided `fly_wing_escape` with `farm_goal=afford_fly_wing`,
      reason "job change ASAP maximizes efficiency; every novice kill is wasted
      progression". Committed `862e12f4b`.
- [x] Add a `farm_goal` field to the decision (e.g. "afford_fly_wing" vs "level_up")
      so the executor knows the farm is short-term. Persisted to store.
- [x] Verify the executor reads `farm_goal` (or the strategy) to farm briefly then
      job change — BOTH emitters (heuristic JOB_CHANGE handler + progression
      domain) now resume job change once zeny affords the escape.
- [x] Full test suite passes (463 passed, 3 pre-existing cold-start failures)
- [x] Commit + push — `862e12f4b`
