# COMPLETENESS — RULE.md Violation Fix (2026-09-03)

## Goal
Fix ALL RULE.md violations found in the deep audit. Reconcile, not trim. The
LLM/AI agent (conscious tier) must decide strategy; reflex only executes; no
server-specific literals in *.py; gear/consumable decisions agent-driven.

## Violations found + disposition

### CRITICAL — FIXED
- [x] V1: death-loop reflex (pdca_loop.py:2219) hardcoded `prt_fild01`/`prt_fild05`
      to override the LLM's job-change walk. RE-HOMED: the reflex now reads the
      LLM-decided `survival_strategy` from the server_solutions store
      (level_up_first / job_change_now / fly_wing_escape) and only EXECUTES it.
      Added `_llm_survival_advisory` (conscious tier) that decides + persists the
      strategy. Committed `59f8994d1`.
- [x] V2: heuristic_service.py:2002 hardcoded map-prefix gate
      (`prt_fild`/`pay_fild`/`gef_fild`) gating TacticsDispatcher. REPLACED with
      agnostic `not _is_city_map(_map)` — tactics run on any field map on any
      server. Committed `59f8994d1`.

### NOT VIOLATIONS (allowed game constants per RULE.md §9/§11)
- [x] V3: potion IDs (501/502/504/569) in situational.py/recovery.py — §9
      explicitly allows hardcoding item IDs (Red Potion 501). NOT a violation.
- [x] V4: Fly Wing/Butterfly Wing IDs (601/602) — §9 allows item IDs; 601 IS the
      standard rAthena Fly Wing. The 715 confusion was the empty-inventory bug,
      not a wrong ID. NOT a violation.
- [x] V5: per-class config audit (swordman/thief/acolyte/archer/mage) — §11
      explicitly requires per-class config. NOT a violation.

## Verification
- [x] Full test suite: 463 passed, 3 pre-existing cold-start failures (fail on
      clean stash too — NOT from these fixes)
- [x] Committed + pushed `59f8994d1`
