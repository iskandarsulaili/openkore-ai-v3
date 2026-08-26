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
- [2026-08-26] RC1 fixed (llm/config.py fallback) + sidecar restarted + LLM brain
  verified alive. UNCOMMITTED. Started campaign; created this checklist.
