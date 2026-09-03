# COMPLETENESS — RULE.md Violation Fix (2026-09-03)

## Goal
Fix ALL RULE.md violations found in the deep audit. Reconcile, not trim. The
LLM/AI agent (conscious tier) must decide strategy; reflex only executes; no
server-specific literals in *.py; gear/consumable decisions agent-driven.

## Violations to fix (from audit)

### CRITICAL
- [ ] V1: death-loop reflex (pdca_loop.py:2219) hardcodes `prt_fild01`/`prt_fild05`
      — re-home to LLM-decided `survival_strategy` from server_solutions store.
      Reflex only executes the conscious decision.
- [ ] V2: heuristic_service.py:2002 hardcoded map-prefix gate
      (`prt_fild`/`pay_fild`/`gef_fild`) gating TacticsDispatcher — remove/DB-back.
- [ ] V3: hardcoded potion/gear tables in heuristic_service.py (159-161, 510, 1553),
      situational.py (43-46, 191), recovery.py (15-27) — move to DB-backed lookups.
- [ ] V4: Fly Wing/Butterfly Wing IDs hardcoded (601/602) in nav_engine.py:31-32,
      travel_recommender.py:31-33 — DB-resolve (server-specific).

### MEDIUM
- [ ] V5: per-class config audit hardcoded in heuristic_service.py:5812-5820
      (swordman/thief/acolyte/archer/mage) — DB/agent-driven.

## Verification
- [ ] Each fix: unit test + live check
- [ ] Full test suite passes
- [ ] Commit + push each batch
