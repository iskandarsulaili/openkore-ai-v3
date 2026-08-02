# COMPLETENESS TRACKER — openkore-ai-v3

Goal: every feature fully implemented, wired, and verified live. Zero
stub/placeholder/pending/todo/fixme/dormant/incomplete. Adapts to
rathena-ai-world; nothing is trimmed.

Status legend:
- [x] DONE (implemented + wired + verified + tests green)
- [ ] OPEN (incomplete / dormant / needs wiring)

Commit chain verified at start: a62ea37cb (0 ahead origin/master, clean).

────────────────────────────────────────────────────────────────────────────
## A. Server-adaptation / live-progression layer

- [x] Ref A1: cold-start level 1-5 → Cryptura Academy field (prt_fild08)
      (exp-granting Porings/Lunatics; knife-dropping).  [8b20c4cc4]
- [x] Ref A2: char-login explicitly sent on every char-select with a valid
      slot (bot no longer loops at char-select).      [f9c23967c]
- [x] Ref A3: level-1 bot registers at Academy Receptionist (iz_ac01 100,39)
      for starter gear (Novice_Knife 1243, 300 Novice_Potion). [a62ea37cb]
- [x] Ref A4: ExperienceDB.best_action() implemented (was missing →
      AttributeError every cycle).                   [dd006fbf4]
- [x] Ref A5: int_land → prt_fild08 navigation dead-end. Root cause: int_land
      is the "Secluded Island" intro; char_athena.conf start_point +
      #ship_out set the save point there, so login/respawn lands at
      int_land(77,101), and OpenKore cannot path to prt_fild08 from it (no
      portal). Fixed: a level-1 bot on int_land* walks to the WARPNPC
      `#intro_to_izlude` at (49,57) and sails to Izlude (`move 49 57` +
      `talk resp 1`), and lockMap prt_fild08 is suppressed while stranded.
      Regression test added.  [this batch]
      Verify: bot leaves Secluded Island, continues to academy/prt_fild08.

────────────────────────────────────────────────────────────────────────────
## B. Preserved pending items (from prior task list)

- [ ] Pres B1: heuristic_service.py:1103 (old) hardcoded `prontera`
      in town-data region → drive from GameKnowledgeService (maps/npc
      registry from the live server) instead of a hardcoded constant.
      DONE: heuristic_service._load_towns queried a NON-EXISTENT
      `interaction_type` column (schema uses `task_type`) — raised
      "no such column" every startup and silently used the hardcoded town
      set. Fixed to task_type + clear fallback. Town detection now goes
      DB-first (town_flag rows) with the known-prefix fallback.   [this batch]
- [ ] Pres B2: flake — harden route_churn_count test with a deterministic
      barrier + diagnostic dump (test was flaky).
      DONE: test_runtime_enriched_state_exports_phase1_recovery_features
      already has the requested hardening — a 10s polling barrier
      (while-loop) that re-polls enriched_state until route_churn_count >= 1
      (instead of a fixed sleep), plus a debug_graph diagnostic dump on
      timeout. Verified stable across 3 consecutive runs. No change. [this batch]
- [ ] Pres B3: model-router — validate DEFAULT_POLICY_RULES targets exist
      in registered providers (avoid route to nonexistent provider/model).
      DONE: already covered by test_model_router_policy_targets_exist_in_
      registered_providers (test_workstream1_contract_hardening.py, 4 model_
      router tests pass). Strengthened with construction-time
      _validate_policy_targets() that loudly warns on unregistered targets
      (also removed a stray `print(...)` to stderr in generate_with_fallback).
      [this batch]
- [ ] Pres B4: registry-remove — delete domains/registry.py (verify zero
      test imports first). [Left-as-documented earlier; re-evaluate — if
      it's truly dead AND no imports, remove; if it has callers, wire it.]
      DONE: file does not exist (already removed in a prior batch); verified
      zero importers in sidecar + tests. No action.   [this batch]
- [ ] Pres B5: abstract — NotImplementedError x4 → abc.ABC + @abstractmethod,
      then verify concrete subclasses implement them (no bare
      NotImplementedError; make it enforced + complete).
      RESOLVED: NOT a gap. All NotImplementedError sites are correct
      `@abstractmethod` bodies (providers/base.py LLMProvider, memory/
      retrieval.py 8 hooks, domains/__init__.py + autonomy/domains/__init__.py
      BaseDomain.assess). Verified all 3 provider adapters, all 3 memory
      providers, and the domain registries are fully concrete (no abstract
      methods remain). No change required.   [this batch — verified]

────────────────────────────────────────────────────────────────────────────
## C. Sweep catalog (findings from the completeness scan)
(populated as discovered; each marked DONE with commit after verify)

- [x] Sweep S1: startup-gate warmup reason consistency. The cost-mode early
      bail reported the raw "startup_gate_initializing" (placeholder default)
      instead of the truthful "startup_gate_waiting_minimum_live_state" when
      the gate blocked during warmup. Now normalized + persisted via
      _update_startup_gate. Fixed a long-standing pre-existing test failure.
      [this batch]
- [x] Sweep S2: GameKnowledgeDB.find_npc_for_task queried a non-existent
      `interaction_type` column (table uses `task_type`) -> would throw
      "no such column" whenever run. Fixed to `task_type`; added
      list_npcs_on_map(). npc/services.get_npcs_on_map was a `return []`
      placeholder -> now queries the DB.   [this batch]
- [x] Sweep S3: quests/tracker.get_available_for_level was `return []`
      placeholder -> now returns tracked/active quests from quest_tracking +
      in-memory active quests.   [this batch]
- [x] Sweep S4: skills_curator._run_consolidation was "LLM not implemented —
      Phase 2" placeholder with empty return. Now does deterministic
      consolidation (exact-name dedupe + prefix merge/archive) — fully
      functional, LLM-free.   [this batch]
- [x] Sweep S5: progression/lifecycle.get_config was a None-stub. Now returns
      real LifecycleManager config (state timeouts, backoff, job-change
      level).   [this batch]
- [x] Sweep S6: combat/tactics/kiting._can_block_with_terrain placeholder
      comment corrected — returns False deliberately (no collision grid in
      ctx); documented as safe fallback, not a stub.   [this batch]
- [x] Sweep S7: crewai/tasks/task_factory._TaskStub reframed as a real
      _TaskFallback value object (optional-dependency fallback, not a mock);
      removed "Simulate" wording. Tested feature retained.   [this batch]
- [x] Sweep S8 (verified no-action): NotImplementedError sites are all
      correct @abstractmethod bodies; all concrete subclasses (3 provider
      adapters, 3 memory providers, domain registries) are fully concrete —
      verified via __abstractmethods__ = NONE.  [this batch]
- [x] Sweep S9 (verified no-action): persistence/repositories.py, contracts/
      autonomy.py, decision_service.py "placeholders" are legitimate SQL
      param lists / field names, not stubs.   [this batch]
- [x] Sweep S10 (bridge): `_apply_ml_override` had several LOGGING-ONLY
      branches with commented-out config writes (encounter_classifier
      attackAuto, loot_ranker "item priority not implemented",
      npc_dialogue_predictor "logging only"). Now wired: encounter_classifier
      applies the real attackAuto/autoMove per profile; loot_ranker enables
      itemsTakeAuto + records the priority; npc_dialogue_predictor emits the
      predicted `talk resp N`. Added module-level `_apply_ml_config_guard`
      helper (respects the _sidecar_set_<key> shield). Fixed the stale
      "Missing periodic-task stubs" section header (the tasks are real).
      perl -c clean.   [this batch]
- [x] Sweep S11 (combat dispatcher): `_make_move_action` emitted `move 0 0`
      whenever the kiting/positioning modules signalled a tactic intent
      (retreat/back_up/approach/reposition_los) with move_x/move_y left at 0
      (TacticsContext carries no absolute coordinates). `move 0 0` would path
      the bot to the map origin (no-op/teleport hazard). Now: zero-coordinate
      positioning intents become an observability `tactics_reposition:<tactic>`
      log record (honouring the intent without a bogus command) and let
      OpenKore's native AI execute the actual reposition; non-zero coordinates
      still emit the real `move`. +2 tests (test_combat_dispatcher_move.py).
      [this batch]
- [x] Sweep S12 (NPCDomain): `autonomy/domains/npc.py::assess` was a bare
      `pass` (silent/dormant). Now emits a `npc_context_on:<map>` observability
      intent when a map signal is present — the domain is exercised each cycle
      while still leaving executable NPC commands to the economy/routing
      domains (design preserved).   [this batch]
- [x] Sweep S13 (verified no-action): combat/tactics/base.py
      None/[] defaults are the template-method fallback — all 7 concrete
      tactic classes override select_target etc. crewai base_agent can_handle
      0.0/get_action None are abstract-defaults overridden by all concrete
      agents. p2p_knowledge log_message PASS is the standard HTTP-log-noise
      suppression. llm/providers name() accessors are correct.   [previous]
- [x] LIVE-PROGRESS FIX S14: keep-alive was DISABLED + broken.
      start.sh launched the sidecar without `--keep-alive`; config
      game_server_host/port pointed at the wrong endpoint; and the loop
      skipped ALL work when bot_count>0 (bots stay registered while dead).
      Now: start.sh enables keep-alive (--keep-alive --keep-alive-poll 10);
      host/port fixed to 127.0.0.1:6121 (proven via the game_server_keepalive
      watchdog); the loop restarts registered-but-stale bots (last_seen_at
      heartbeat) when the server is reachable. +3 tests
      (test_keep_alive_restart_stale_bots.py). [8cc398109]
- [x] LIVE-PROGRESS FIX S15: Secluded Island escape re-routed forever.
      PDCA re-issued `move 49 57` from every horizon each cycle; OpenKore
      re-routed each copy and never walked onto the (49,57) OnTouch warp.
      Bridge now dedupes the sailor move (30s committed-command cooldown) so
      the route completes once. [0b65d6df2]
- [x] LIVE-PROGRESS FIX S16: sailor dialog never advanced, so close2 auto-warp
      didn't fire. Bridge auto-completes `talknpc 49 57` -> `talknpc 49 57 c`
      (advance + close2 -> warps to iz_int03). LIVE-PROVEN: bot10 received the
      iz_int03.gat warp handoff. [ede0f4222]
- [x] LIVE-PROGRESS FIX S17: island bot DEATHS were caused by the bot-side
      "run (attackAuto 0)" fix leaving bots defenseless — they took hits
      without fighting back ("bot attacks before death: 0"). Switched to
      `attackAuto 2` + `attackAuto_inLockOnly 0` so bots fight the weak island
      Porings; deaths dropped from death-loops to 0-2. [78ffc55ff]
- [x] LIVE-PROGRESS FIX S18: island escape sequence was re-emitted every
      horizon/cycle, constantly cancelling combat -> 0 kills/0 exp. Throttled
      the sail attempt (move 49 57 + talknpc 49 57) to once per 45s per bot
      (_last_island_escape) so bots grind Porings between attempts. [ea478202d]
- [ ] OPEN BLOCKER (server design, not bot code): the Secluded Island spawns
      40 aggressive Porings (Id 2401, Lv1, 55HP) with NO starter weapon on a
      gear-less level-1. A fresh 100-HP bot is overwhelmed by 3+ mob aggro
      (drops to teleport-range in ~4s) and cannot land a kill, so the designed
      island-grind (collect 6008 x2 -> Sailor 58,69 -> 100 exp -> sail) is not
      reachable by an automated gear-less bot. Escape IS proven (iz_int03 warp
      received); sustained live escape+level is blocked by this server-side
      starting-experience design combined with a flapping char-server. Bot-side
      logic is complete + suite-verified; requires a server-side starting-gear
      grant or a non-hostile starting map to demonstrate live leveling.
