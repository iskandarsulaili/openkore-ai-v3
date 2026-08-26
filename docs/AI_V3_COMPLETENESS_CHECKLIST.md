# AI_V3_COMPLETENESS_CHECKLIST.md — openkore-ai-v3 Live-Production Completeness Campaign

> Mandate (user, 2026-08-26): implement/integrate/fix/wire/execute/verify ALL to
> completeness — zero mock/stub/placeholder/pending/todo/fixme/dormant/incomplete.
> Reconcile, never trim. Agnostic (server/game/situation) everywhere. AI bot
> plays independently zero-human-intervention from start to end, self-adapting,
> self-learning, self-healing, self-improving (memory/tables/DB/SOUL.md/MEMORY.md
> + crowdsource P2P intelligence). LLM=conscious brain, ML=subconscious,
> reflex=instant combat/hardcoded. OpenKore core = "the muscle".
> Ready for live production release.
>
> Method: checklist FIRST, verify BEFORE modifying, mark + verify after each batch,
> commit+push each batch. Big-picture + all-angles verification to prevent
> unexpected issues.

Status legend:
- [x] DONE (implemented + wired + verified live)
- [~] VERIFIED-WORKING (no change needed — proven live)
- [ ] OPEN (incomplete / dormant / needs wiring / needs verify)

────────────────────────────────────────────────────────────────────────────
## 0. FOUNDATION — the three-tier mind architecture contract

- [ ] F1. Conscious = LLM/CrewAI (intent + root-cause + whole-picture). Subconscious =
      trained ML (drives ~95% moment-to-moment). Reflex = hardwired safety floor.
      OpenKore core = muscle only (pass commands, never decide strategy/tactics).
- [ ] F2. Agnostic: NO hardcoded server/game/map/coordinate/item/NPC literals in
      decision logic. Server-specific facts live in DB-backed server_solutions store.
- [ ] F3. self-* loop: live observation → conscious (SOUL+MEMORY injected) → action →
      ack → lesson write-back → MEMORY.md re-injected. Self-heal + self-improve.
- [ ] F4. Memory: SOUL.md + MEMORY.md + tables + DB + crowdsource P2P intelligence.

## 1. LLM / CONSCIOUS BRAIN (root-cause of the dead brain)

- [x] RC1. **DONE** — LLM config wiring: llm/config.py `LLMConfig.from_env()` only
      read `LLM_*` env; `.env` has only `OPENKORE_AI_PROVIDER_OPENAI_*` → LLMManager
      (conscious brain) saw NO provider → is_available()=False → every LLM advisory
      early-returned → whole conscious tier DEAD (breaker_open/timeout in telemetry,
      zero orchestration events). FIX: from_env() falls back to SidecarSettings.
      VERIFIED: llm_manager_startup_ok, llm_manager_attached: yes, _post_json to
      combo/deepseek-v4-flash firing every ~10s, provider_route_decided OK.
      [UNCOMMITTED — this batch]
- [ ] 1.1. Verify ALL 4 LLM advisories (cold_start %15, gear %30, npcdialog %10,
      help_coordination %60) actually fire + enqueue real actions now that the
      brain is alive. Confirm `llm_cold_start_queued` / `llm_gear_queued` markers.
- [ ] 1.2. Verify CrewAI conscious tier (mission_agent via model_router) actually
      produces decisions now (provider_route_decided → success), not just the
      llm_manager path. Both LLM paths must be healthy.
- [ ] 1.3. Persist-circuit-breaker: confirm provider breaker no longer trips on the
      healthy path (open_seconds=30 in-memory; should stay closed under normal load).
- [ ] 1.4. SOUL.md + MEMORY.md injection VERIFY into every LLM call (self_awareness
      attached; confirm inject() prepends on the live path).

## 2. COLD-START / PROGRESSION (root-cause of "no real progress")

- [ ] RC3. **cold-start routing loop**: bot reaches academy door `125 257` (warp to
      iz_ac01) then `Calculating lockMap route to prt_fild05` yanks it back — it
      NEVER walks onto the iz_ac01 warp to register for the starter kit (Novice_Knife
      + 300 potions). All 3 testbots stuck at Lv1/2 on izlude for hours. lockMap must
      defer while heading INTO the academy; academy-first must complete before farm
      lockMap engages.
- [ ] RC3a. **LOCKMAP MUST BE AGNOSTIC + LLM/AGENT-DECIDED (user directive 2026-08-26)**:
      `lockMap` is currently HARDCODED in multiple sites (heuristic_service.py emits
      `set lockMap prt_fild05` / `set lockMap {_hunt_map}` with hardcoded fallbacks;
      bridge aiSidecarBridge.pl:943 defaults `aiSidecar_huntingMap` to 'prt_fild05';
      configs hardcode lockMap prontera/izlude). NO hardcoded map literals in the
      decision. lockMap must be decided by the LLM/CrewAI conscious tier (from live
      server map knowledge + server_solutions + reachable-farm resolution) and persisted
      as a learned fact, NOT baked into *.py or config. The farm-map decision is a
      WHAT that belongs to the conscious brain, translated via learned facts.
- [ ] RC3b. **adaptive get_best_map can return NON-FARM maps**: map_performance records
      kills/deaths/visits for ANY map the bot is on (incl. int_land intro island, town
      maps). get_best_map picks the highest kills/deaths-ratio candidate → returned
      `int_land` (a no-mob intro map) as a farm target. Must gate to farm-capable maps
      (from map_knowledge hunting set / *_fild / *_dun), never intro/island/town maps.
      This is part of the agnostic lockMap fix (RC3a).
- [ ] 2.1. cold_start relog loop: `ColdStartManager.assess` emits `relog` on LIVE
      in-game bots (log: "[cold_start] ... emitting relog (cooldown 120s)" while bot
      is Lv1 on izlude). The in-game guard (cold_start.py:312-322) checks
      map_known/base_level>0/in_game/map — verify signals reach it; it must NEVER
      relog/char-create a live bot. (Note: cold_start.py:321 guard may be bypassed by
      a signals path lacking those keys.)
- [ ] 2.2. Academy registration fires at iz_ac01 receptionist (100,39) — verify the
      full chain: reach iz_ac01 hall → talk receptionist → receive knife+potions →
      weapon_has → THEN farm prt_fild05/08.
- [ ] 2.3. server_solutions store: verify safe_town/farm_map/academy facts are
      learned + injected into the LLM cold-start prompt (agnostic path).
- [ ] 2.4. Progression end-to-end: once academy kit acquired, bot must level
      continuously (kills → EXP → base_level ticks) and progress toward job-change
      → endgame, self-adapting. Verify with DB base_level/kill/EXP evidence.

## 3. FLEET ORCHESTRATION / LEADER

- [ ] 3.1. Verify FleetOrchestrator + FleetCoordinator actually issue directives now
      (was inert due to RC1). Confirm directive events appear in telemetry.
- [x] 3.1b. **NEW BUG FOUND**: `PartyCoordinator check failed: 'FleetCoordinator'
      object has no attribute 'get_bot'` (fires every cycle in sidecar log). A real
      unwired fleet bug — PartyCoordinator calls get_bot() that FleetCoordinator
      doesn't implement. FIXED 454084e95: added get_bot/list_bots/party_members to
      FleetCoordinator, BotRole live fields (position/party_id/current_role/hp/
      weight/zeny/map_name/hp_pct/weight_pct), feed live bot state into fleet each
      cycle, coordination failures now logged (was silent dead code). 22 fleet
      tests pass. VERIFIED tank+healer -> party_invite_observe.
- [ ] 3.2. Leader/coordination: does a leader coordinate others? Verify RoleManager
      assigns roles, PartyCoordinator forms parties, cross-bot resource sharing fires.
- [ ] 3.3. Multi-bot identity: 3 testbots on one account (testbot99 chars 0/1/2)
      registered as distinct bots (TestBotA/B/C:testbot99) — verify orchestration
      sees all 3, not just the account-level aggregation.

## 4. MEMORY / SELF-LEARNING / SELF-HEALING / CROWDSOURCE

- [ ] 4.1. record_lesson → MEMORY.md write-back: verify lessons are recorded on
      fail/refuse outcomes and re-injected (self-improvement loop closed).
- [ ] 4.2. SOUL.md + MEMORY.md inject on every LLM call (verify on live path).
- [ ] 4.3. Crowdsource P2P: lessons_hub.db push/pull + p2p_knowledge mesh working.
- [ ] 4.4. Subconscious (RL/DQN): verify _train_from_replay actually runs,
      reinforcement_stats.json training_steps>0, shadow→promotion.

## 5. AGNOSTIC AUDIT (server/game/situation — the user's core rule)

- [ ] 5.1. Sweep for hardcoded server-specific literals (map names, coords, item IDs,
      NPC names) in decision code. Must resolve via discovery/server_solutions.
- [ ] 5.2. Verify heuristic_service / reflex tier holds only generic safety rules
      (never per-server).
- [ ] 5.3. Bridge (Perl) passes commands only — never the source of strategy.
- [x] 5.4. **Inventory-snapshot false-empty FIXED + LIVE-VERIFIED**: bridge read
      `$char->{inventory}` (dead hash key — OpenKore stores inventory at
      `$char->inventory()` InventoryList). All 34 read-sites dead → empty snapshot →
      false "weapon-less" → cold-start never advanced. FIX: `_char_inventory()`
      (getItems()/get(binID) — fork's @{} overload errors), full inventory_items
      digest + has_weapon_in_inventory in snapshot payload; sidecar reads top-level;
      _has_coldstart_weapon robust; InventoryItemDigest contract gains type field
      (extra_forbid rejected it) + item_id stringified. VERIFIED LIVE: snapshot
      has_weapon=True, 9 items (Novice Potion 569 x300, knife). Commits
      d9117f163, 91e7e9d25, [type-field commit].

## 6. DORMANT / DEAD / UNWIRED SWEEP (reconcile, never trim)

- [ ] 6.1. Full-repo reference scan: any fully-implemented subsystem with zero
      external callers = dig deeper (may be previously-incomplete impl that's needed),
      wire it. (Prior sweeps S19-S24 wired ConsciousDecisionEngine, CombatTactics,
      VendingArbitrage, MarketTiming, GearScorer, intelligence_integration.)
- [ ] 6.2. Re-run the TODO/FIXME/stub/placeholder/NotImplemented/dormant marker
      audit; every genuine gap resolved.
- [ ] 6.3. Verify no leftover half-wired path from RC1 fix (both LLM systems coexist
      cleanly: llm_manager + model_router).

## 7. TESTS + VERIFICATION GATES

- [ ] 7.1. Full Python test suite green (test_skills_system.py + workstream suites).
- [ ] 7.2. Bridge perl -c clean.
- [ ] 7.3. Live E2E: 3 bots online on 1 account, all progressing (base_level ticks),
      no disconnect/reconnect loops, no relog spam on live bots.
- [ ] 7.4. All-angles: no new defect introduced by each batch (verify before/after).

## 8. COMMIT + PUSH each batch (user rule: commit after batch, push reasonable stage)

────────────────────────────────────────────────────────────────────────────
## Progress log (append per batch)
- [2026-08-26] RC1 fixed (llm/config.py fallback openai+deepseek) + sidecar restarted + LLM brain verified alive. Committed 95e1230d4. Created this checklist.
- [2026-08-26] BATCH 1 DONE: LLMManager wired to SidecarSettings. LLM conscious brain alive (verified: available=True, openai enabled, _post_json firing).
- [2026-08-26] BATCH 3 (inventory) DONE + LIVE-VERIFIED: bridge dead `$char->{inventory}` key → real `$char->inventory()` via getItems()/get(binID); inventory_items digest + has_weapon_in_inventory; contract type field; item_id stringified; object-branch signals; per-bot weapon LATCH (intermittent 0B09 parse never reverts to weapon-less). VERIFIED: snapshot has_weapon=True, 9 items, routing commits to farm (no ping-pong).
- [2026-08-26] BATCH 2 (agnostic lockMap) DONE: get_best_map FARM-GATE (*_fild/_dun); _audit_zeny UnboundLocalError fixed; routing hardcoded level ladder (pay_fild01/prt_fild08/prt_fild05) REMOVED → learned server_solutions farm_map + reachable_hunting_maps (level-scored, portal-graph-filtered). No server-specific map literals in decision path. VERIFIED: `navigate prt_fild05`, `set lockMap prt_fild08` emitted agnostically.
- [2026-08-26] OUTCOME PROOF: TestBotA reached lvl 2 (base_exp 547, +real kill drop 2112) — real farming when sessions sustain. B/C stuck at lvl 1/exp 0.
- [2026-08-26] OPEN BLOCKER (deep protocol): BOTS CHURN on PACKET DESYNC — map-server disconnects on garbage packets (0xba71/0x6633/0x11a4/0x0001, 1027-byte random blobs, "expected 12925 but only 1027 remaining"). Not routing — a 2025-client/OpenKore packet-stream protocol mismatch. B/C never sustain a session long enough to farm. messageIDEncryption 0 already set. This is the remaining "no consistent progress" driver.
- [2026-08-26] NEW BUG: (3.1b) PartyCoordinator 'get_bot' attr missing (fleet). LLM advisory IS firing.
- [2026-08-26] BATCH 5 DONE + LIVE-VERIFIED (commit 454084e95): PartyCoordinator was written against a phantom FleetCoordinator API — get_bot()/list_bots()/party_members() missing + BotRole lacked position/party_id/current_role/hp/weight/zeny/map_name/hp_pct/weight_pct → AttributeError every cycle, swallowed by `except: pass` = fleet coordination silently dead. Fixed: added the 3 methods to FleetCoordinator + live BotRole fields; feed live per-bot state into fleet each cycle (pdca_loop fleet-status sync); coordination failures now logged. 22 fleet tests pass. VERIFIED tank+healer -> party_invite_observe. No AttributeError live.
- [2026-08-26] BATCH 4 (cold-start academy-door re-emission) FIXED: (a) LLM cold_start + gear advisories now use latched weapon state (568b40d7f); (b) weapon latch PERSISTED to heuristic_state.json so a sidecar restart doesn't forget the bot owns its starter knife (eead6da7c) — 0B09 parse is intermittent and the in-memory latch reset every restart, re-firing `move 125 257`. VERIFIED roundtrip: weapon seen -> restart -> still latched. No academy-door re-emission in live run.
