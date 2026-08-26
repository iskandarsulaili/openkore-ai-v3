# AI_V3_COMPLETENESS — BATCH 6: Dormant/Unwired Sweep + Agnostic Audit + Tests Green

Date: 2026-08-27
Goal: systematically find every dormant/dead/unwired module, every stub/mock/TODO/pass/placeholder,
every hardcoded server-specific value in the decision path, and every failing/wiring-gap test.
Reconcile-not-trim (never remove features). Verify each fix live where possible.

## Scope
- [ ] 6.1 Run FULL test suite baseline (all 63 test files) — identify failures/errors first.
- [ ] 6.2 Sweep for dormant modules: defined-but-never-imported / called-but-never-wired.
- [ ] 6.3 Sweep for mock/stub/TODO/FIXME/pass/placeholder/not-implemented in source.
- [ ] 6.4 Agnostic audit: any hardcoded map/item/coord/mob/server literal in DECISION path (not knowledge tables).
- [ ] 6.5 Verify live wiring: LLM brain, fleet, cold_start, routing, combat, economy all reachable from pdca_loop.
- [ ] 6.6 Tests green (all pass).
- [ ] 6.7 Commit + push.

## Findings (append per round)
- [2026-08-27] 6.1 FULL TEST BASELINE: 408 passed (406 + macro compiler 2). All green.
- [2026-08-27] 6.2 DORMANT MODULE FOUND + FIXED: `domains/combat/combat_intel.py` (wires 4 self-learning PVP modules: GTB detection, elemental armor checker, class counters, hit/flee analyzer) was NEVER called — lifecycle passed `combat_intel=None`, assess_combat_intel unreferenced in the assess chain. WIRED into `combat/dispatcher.py` assess() Phase 6 (lazy import, no cycle). Now produces real PVP actions each combat tick.
- [2026-08-27] 6.3 Abstract `NotImplementedError`s are all legit @abstractmethod base contracts (providers/base.py, memory/retrieval.py, domains/__init__.py) — not stubs.
- [2026-08-27] 6.4 Legacy domains (combat/consumables/economy/...) are INTENTIONAL observe-only (test_legacy_domains_observe_only.py documents: double-emission + party-spam hazard, fixed at source) — not dormant. Verfied achievements.py wired via knowledge_ingestion.py + personality_engine.py.
- [2026-08-27] 6.4 AGNOSTIC AUDIT — removed REMAINING hardcoded map literals in LIVE heuristic_service._assess_impl decision path: (a) farm-target ladder (prt_fild05/pay_dun00/orcsdun01) → learned server_solutions farm_map (reachability-checked) → reachable_hunting_maps → adaptive get_best_map; (b) `move prontera` town fallback → learned safe_town → derived town → true-last-resort prontera; (c) _FARMS_OK hardcoded set → portal-graph hunting-map resolver; (d) raw weapon recompute → latched _has_coldstart_weapon. Fixed tuple-unpack bugs (_reach[0][0], get_hunting_maps returns list[tuple]). 33 tests green. Committed f09ba5291.
- [2026-08-27] 6.4 AGNOSTIC AUDIT (round 2): step-7 town + step-8 post-job hunt now agnostic via shared _resolve_safe_town (learned safe_town → DB town set → prontera) + GameKnowledgeDB.get_hunting_zone(class_hint=job) + reachable resolver. Removed hardcoded 'move prontera' + job->map dict. 39 tests green. Committed f29db1d85.
- [2026-08-27] 6.6 PER_MAP_MON_CONTROL (ro_mechanics.py) = STATIC KNOWLEDGE REFERENCE (map→mob spawn facts), same class as MAP_KNOWLEDGE (deemed acceptable knowledge base, not a decision literal). The live attack decision is DB-gated via MonsterDB (F3: downgrade to ignore if boss/too-strong). NOT a violation; left intact (no-trim).
- [2026-08-27] 6.2 DORMANT MODULE #2: autonomy/domains/progression.py is a DEAD DUPLICATE (0 importers — the live path is domains/progression/cold_start.py). Left on disk per no-trim mandate.



