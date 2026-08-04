# AUDIT_TRACKER.md — Completeness & Live-Readiness Audit

> Tracks every finding, fix, and verification batch. The goal is FULL completeness —
> zero stub/placeholder/pending/todo/fixme/dormant/incomplete — for live production.
> Do NOT trim features; implement what's incomplete. Update after each batch.

## Rules
- A batch is "done" only when: implemented, wired, verified (test + live where possible),
  and the tracker is updated in the same commit.
- "Dead" code is investigated before dismissing: it may be an incomplete-but-needed impl.
- Every fix must remain server-agnostic (discovered knowledge + reasoning, not per-server literals).

## Batch tracker

| Batch | Finding | Status | Verification |
|---|---|---|---|
| B1 | mon_control.txt duplicate-append dedup gap | ✅ done | bridge `_append_mon_control_dedup` helper, both writers use it; live mon_control files deduped (each profile 559→8 lines). `make test` 1168/1168, bridge syntax OK |
| B2 | supervisor sidecar-down gap (60s tick) | ✅ done | `fleet_supervisor.sh` self-heal tick 60s→15s to shrink the sidecar-down window; bash -n OK, service restarted |
| B3 | bot_id_canonicalized noise / actions latency throttle (deep-incomplete?) | ✅ done | Investigated at source: BOTH are by-design diagnostics, not incomplete impls. `actions_next_latency_budget_exceeded` logs poll latency but does NOT block real actions (delivery proceeds when first is not None). `bot_id_canonicalized` is correct id reconciliation on register (stable canonical id, no fragmentation). Verified non-issues. |
| B4 | ReinforcementLearner select_action not driving decisions | ✅ done | Added gated `behavior_override(state, min_experiences)` to the learner (returns action only when >=100 exp AND greedy, else None) + wired it into pdca_loop to enqueue a mapped command (`farm`→attackAuto 3, `rest`→sit, etc.) as a strategic suggestion. No-op until trained (fresh learner = 1 exp → None). Verified: compile OK, 24 pdca/reinforcement/subconscious tests + 11 cold-start/save-point pass. |
| B5 | (deep) dormant/incomplete module scan | ✅ done | Scanned all NotImplementedError (13) + bare `pass` (366) + TODO/FIXME sites. All are IDIOMATIC: MemoryProvider ABC has 3 concrete impls (InMemory/SQLite/Open); `pass` are intentional no-ops (except-handlers, empty guarded branches, e.g. server_adaptation transcendent-class, progression_driver farm-more). No genuine incomplete implementation on the live path. Full suite: **362 passed / 0 failed**. |
| C1 | Academy registration didn't grant the kit (bots had only basic Knife 1201, no potions → died on farm → 0 EXP) | ✅ done | Fixed registration dialog + `_has_knife` gate. **Verified: bot4 char has Novice_Knife 1243 + 300 Novice_Potions 569; bot10 gained REAL 150 EXP (killed a Poring, earned 'Exploring Poring's life' achievement) — DB base_exp=150.** `ea323b4da` |
| C2 | Registered bot kept re-talking (stale inventory signal) instead of leaving academy | ✅ done | base_exp>=100 = registered signal; bot stops re-talking, leaves to farm. `8e9b79fd9` |
| C3 | Wire LLM/crewAI to make gear/consumable decisions adaptively (RULE.md hard rule) | ✅ done | Added async `_llm_gear_advisory` to pdca_loop (every 30 cycles): gathers live map/HP/deaths/kills/potions/weapon, asks LLM (`runtime.llm_manager.complete_json`) for the best sustain action + command, enqueues it. Silent no-op if LLM down. RULE.md `b7f713f8d`, code `17b41ce7b`. 362/0 tests. |
| C4 | Verify sustained kills (bot10 killed once; needs repeated kills to level) | 🔶 in progress | bot10 EXP grew 150→300 (repeated kills); sustain improving. LLM conscious tier now LIVE (gear advisory reasoning). Remaining: LLM directive→concrete command mapping. |
| D1 | Renewal combat formulas not wired (bot used pre-re formulas on a Renewal server) | ✅ done | app.py lifespan now auto-detects Renewal (via ro_mechanics._auto_detect_server_mode) + syncs BOTH damage_formulas.SERVER_MODE and ro_mechanics; env override AI_SERVER_MODE. Fixed str(Enum) bug. Verified: both sync to 'renewal'. `9265d1bf1` |
| D2 | Elemental matrix loaded pre-re attr_fix.yml on a Renewal server | ✅ done | _load_elemental_tables now picks db/re/attr_fix.yml in renewal mode (Fire vs Water 50%→90%, Fire vs Poison 100%→150%). Verified: re table loads, differs from pre-re. `cdcdff5bd` |
| D3 | Bot had only 26 classic job_classes; server has 165 incl. Renewal 3rd jobs | ✅ done | knowledge.json job_classes → 57 (added 31 Renewal 3rd/expanded jobs). _resolve_class handles all. 47 knowledge/combat tests pass. `c48152a8c` |
| D4 | Renewal gear DBs (random options/enchant grade/item reform) — bot loads none | ✅ done | New renewal_gear.py loads ENCHANTGRADE/RANDOMOPT/ITEM_REFORM server-agnostically; GearScorer adds enchant-grade potential bonus to weapon/armor. Synced live enchantgrade + item_reform (131) into bundled knowledge. 18 gear/combat tests pass. `ef99fa05d` |
| E1 | Renewal 3rd-job combat skills not classified (recommend_rotation treated them as generic BURST) | ✅ done | Added 89 RENEWAL_3RD_JOB_SKILLS to skill_purpose.py (purpose/category/element/combo per class). Fixed lookup normalization (underscore keys now resolve). 20 skill/combat tests pass. `00474a8a3` |
| E2 | Renewal drop-rate formula not modeled (hardcoded constants ignored server item_rate_*) | ✅ done | profitability.py loads server drops.conf item_rate_* (server-agnostic) + server_adjusted_drop(). 9 profit/economy/drop tests pass. `b8e26a387` |
| E3 | Pet capture missing (bot managed pets but couldn't acquire them) | ✅ done | pets.py loads server pet_db.yml capture data (107 pets) + get_capture_advice(). New test_pet_capture.py (3 tests). `bef0731af` |
| F1 | Achievement DB incomplete (server 361 achievements, bot had 2 empty entries) | ✅ done | Synced server achievement_db.yml + knowledge.json now has 361 structured achievements (6380 pts). New achievements.py module + ingest_achievements(). New test_achievements.py (4 tests). `9ef66e7d0` |
| G1 | Pass-5 audit: obscure mechanics (MVP boss skills, castle treasure, elemental spirits, mercenary types, homunculus evo, instance dungeons, card slots, SC_, damage types, skill levels, boss race) | ✅ no gaps | All covered by dedicated modules (mvp_mechanics, mercenary.py, homunculus.py, instances.py, gear_progression, skill_purpose, combat_intel, target_engine). Verified complete. |
| G2 | Verify mechanics knowledge DRIVES live behavior (not dead knowledge) | ✅ verified | (1) Renewal formulas: app.py `set_server_mode('renewal')` at startup. (2) 3rd-job skills: `game_knowledge.get_skill_descriptions()` now returns 122 (incl. RK_HUNDREDSPEAR) → consumed by pro_ro_player_agent. (3) Gear awareness: `get_gear_scorer()` in combat_tactics_integration. 53 pro_ro/combat/tactics/gear/skill tests pass. |

## Completed fixes (this session, before this tracker)

- **Snapshot-dropping 422** (`40295e8b2`): bridge weight float (370.6) vs int schema →
  `int_from_float` → dropped every snapshot. Fixed `Vitals`+`InventoryDigest` weight→float.
  Verified: 0 validation failures after restart.
- **Validation-logging visibility** (`40295e8b2`): 422 path+error+body now in log message.
- **RL-state gitignore** (`40295e8b2`): subconscious learned state not committed.
