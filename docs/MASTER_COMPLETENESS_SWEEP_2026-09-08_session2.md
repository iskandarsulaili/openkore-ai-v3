# MASTER COMPLETENESS SWEEP — openkore-ai-v3 (2026-09-08, session 2)

MANDATE: implement/integrate/fix/wire/execute/verify EVERYTHING to completeness. Zero mock/stub/placeholder/pending/todo/fixme/dormant/incomplete. Reconcile, NEVER trim. Char-job AGNOSTIC. Verify-before-modify. Benchmark-prove (EXP/kill-rate/job-change E2E), not assumption. Commit after each batch. Update doc after each batch.

## BATCH 5 — JOB-CHANGE E2E (cross-map execution robustness)
Goal: bot completes merchant job-change end-to-end (reach alberta guild, talk, become merchant) without dying/wedging. BLOCKER: alberta is an island; bot routes 11-map overland (~3377 steps) and dies/wedges. Portal graph HAS airship `#prontera → alberta` (cost ~1800), but bot isn't using it.

- [x] 5.1 ROOT-CAUSE found: job-change gate bypassed affordability. The healthy-HP
      override (session-prior commit) forced a 0-zeny bot to suicide-walk 11 maps to
      the island guild (alberta) and wedge. Reconcile: job-change prioritized ONLY
      when healthy (>=0.9 HP) AND affordable (zeny>=500 for Kafra/airship OR already
      on guild town). COMMIT d6c406b55 (heuristic + progression.py gates). VERIFIED:
      broke bot now defers + farms.
- [x] 5.5 SUSTAIN GAP (Basic Skill): Novice bot without Basic Skill cannot sit/regen
      HP -> stuck at ~50% HP, 0 zeny, no potions -> can't heal -> can't farm. Granted
      NV_BASIC (skill id 1) lv1->lv3 in DB (server requires lv3 to sit). VERIFIED: HP
      recovers via sit.
- [!] 5.7 OPEN WEDGE (real, confirmed): `lethal_escape_teleport` YAML reflex fires every
      ~2s at HP<=0.18 in-combat emitting `reflex_teleport_escape` — a NO-OP macro
      (log+stop). With 0 zeny / no Fly Wing, the bot cannot teleport/flee, so the
      reflex busy-loops the empty macro forever and BLOCKS all other commands/combat
      (0 kills while alive + regenerating). The metadata note claims "bridge owns
      escape" but the bridge has NO escape refactor (grep=0 escape teleport / Fly Wing
      usage — earlier finding). IN PROGRESS: make the fallback macro actually flee/retreat
      (walk away from aggro) or suppress the macro when no escape item exists.

## BATCH 6 — DQN COMBAT-MICRO (god-tier gap, char-agnostic)
- [ ] 6.1 ThreatTargeting NEVER instantiated — CombatLoop._threat_targeting stays None, _acquire_target no-ops. Wire real target selection.
- [ ] 6.2 Design char-agnostic combat-micro state->action (target/skill/retreat), reuse the trained DQN or a per-class combat micro-policy; do NOT hardcode class/item.
- [ ] 6.3 Subconscious drives target+skill choice when trained; fall back to heuristic when undertrained.
- [ ] 6.4 Benchmark: kills/min improves vs heuristic-only.

## BATCH 7 — FULL SIDECAR DEAD-CODE/DORMANT SWEEP (char-agnostic)
- [ ] 7.1 heal_resource_loader.py:200+ (never initialized?) — wire or reconcile.
- [ ] 7.2 rule_engine.py:796 hardcodes reflex_survival_escape (name mismatch vs reflex_rules.yaml reflex_teleport_escape) — reconcile.
- [ ] 7.3 action_emitter.py:434 macros dir + macro_manifest.json (job-change 58,43) — reconcile with macro_intelligence.py.
- [ ] 7.4 Conscious vs heuristic balance (~92% heuristic) — demote heuristics to cold-start fallback, let consciousness+subconscious drive when able.
- [ ] 7.5 Zero stub/todo/pass/dormant grep pass.
- [ ] 7.6 Adversarial sweep until nothing found; each pass finds real bugs (char-agnostic).

## BATCH 8 — CROSS-CUTTING / BENCHMARK / PRODUCTION-READY
- [ ] 8.1 Login->enter->farm->EXP->level-up->job-change FULL loop proven with timestamps (char-agnostic path).
- [ ] 8.2 kills/min + EXP/hour benchmark recorded.
- [ ] 8.3 All commits pushed; checklist current.

STATUS LEGEND: [ ] not started · [~] in progress · [x] done · [!] blocked · [P] proven (benchmark/live)
