# MASTER COMPLETENESS SWEEP — openkore-ai-v3 (2026-09-07)

MANDATE: implement/integrate/fix/wire/execute/verify EVERYTHING to completeness. Zero mock/stub/placeholder/todo/fixme/dormant/incomplete. Reconcile, NEVER trim features. Verify-before-fix. Benchmark-prove outcome (EXP/kill-rate/level), not intent. Commit after each batch. Update this doc after each batch.

## BATCH 0 — LIVE BLOCKER (gameplay gating)
Goal: bot actually farms end-to-end. This is THE gap between theory and outcome.

- [x] 0.1 ROOT-CAUSE FOUND+PROVEN (map-login barrier): the every-5s keepalive ping in
      aiSidecarBridge.pl fired during LOGIN, coalescing 0B1C (2B) with the 0x0436
      map-login (23B) into a 25-byte segment; map-server (clif_shuffle.hpp expects
      exactly 23) rejected every attempt -> 3.5-day map-login rejection loop.
      FIX: gate keepalive on $mapLoginAcked (in-game only) + import $mapLoginAcked
      into the bridge Globals. PROVEN: bot entered map (02EB Enter Map + combat 08C8)
      at 12:48 after restart. First successful map entry in days.
- [x] 0.1b REVERTED a misdiagnosis: the 23-byte pack branch 'v V V V V V C' was
      CORRECT (I miscounted 6 longs); reverted my false fix immediately.
- [x] 0.2a SECOND BLOCKER (cost/budget) ROOT-CAUSED + FIXED + COMMITTED b6de64645:
      cost_mode=max was configured but the conscious brain was hard-capped at
      100k tokens/day — (a) pdca_loop gate (7173) used raw settings field not
      cost-mode budget; (b) LLM manager _check_daily_budget used unprefixed env
      (standard/100000); (c) lifecycle LLMManager built from LLMConfig.from_env()
      (unprefixed) ignoring OPENKORE_AI_*=max. All three fixed to honor max=unlimited.
      Sidecar restarted (PID 2548302) loading the fixes.
- [x] 0.2b COMMITTED tables/portals.txt pre-existing fix (96072c4bc, user permission).
      Pushed b6de64645..96072c4bc.
- [x] 0.3 THIRD BLOCKER (in-game sustain) ROOT-CAUSED + FIXED + COMMITTED 7138dafd2:
      bot entered map + farmed but HP tanked to 16 with 296 Novice Potions unused.
      (a) the 'no potions -> block' override counted ONLY hardcoded Red/Orange/White
      Potion, so a bot carrying only Novice Potion (569) was judged 'no potions' and
      ALL potion use was silently blocked on hunting maps -> now uses
      _best_available_heal_name() (agnostic real-inventory scan). (b) the
      'use <item>'->'is <item>' rewrite emitted the LOWERCASED name but
      Actor::Item::get does case-SENSITIVE exact match -> 'is novice potion' failed
      to resolve -> now emits the ACTUAL inventory name casing. PROVEN: HP holds
      198/198 (was 16 & dying), EXP gained 886->1271, zero 'does not exist' errors.
- [x] 0.3b FOURTH BLOCKER (farm-map routing loop) ROOT-CAUSED + FIXED + COMMITTED
      e4d1a8961 + b04c49b13 + 2d98830c5: FOUR job-change emitters fought the
      conscious survival_strategy decision. progression.py honored
      level_up_first, but heuristic_service had 3 SEPARATE emitters (job-change
      gate ~3992, cold-start step-7 ~3808, 2-1 ~5605) that re-emitted the guild
      move WITHOUT the gate -> bot oscillated alberta_in/moc_prydb1 <-> farm, EXP
      froze. All 4 gate sites now defer for BOTH 'level_up_first' AND
      'fly_wing_escape' (the LLM's actual decision: farm until it can afford a
      Fly Wing, THEN job change; with 0 zeny it keeps farming). PROVEN: EXP
      1890->7176 continuously, 0 guild dispatches, HP full, in-map sustained.
- [~] 0.3c REMAINING: intermittent disconnects (bot reconnects + resumes farming,
      but drops ~every 10-20 min). Not a routing/sustain bug — reconnect loop
      recovers and EXP keeps climbing. Track for stability.
- [ ] 0.3 ROUTE-FAILURE STALL: 2386 route-calc fails on prt_fild08 (post-stability).
- [ ] 0.4 After 0.1-0.3: one real bot -> continuous EXP farming -> benchmark
      (EXP/hour, kill-rate, deaths/hour, base_level) as definition-of-done.

## BATCH 1 — TOKEN BUDGET (conscious tier gated to actions=0)
- [ ] 1.1 fleet_daily_token_budget_exceeded:106618/100000 → plan emitted but refused. Root-cause the budget mechanics; rebalance so the conscious plan executes without runaway cost.
- [ ] 1.2 Verify each LLM call class uses its purpose-tuned max_tokens/budget_class; no waste.
- [ ] 1.3 Benchmark: measure a full PDCA cycle end-to-end (probe→concscious→action→execute) latency + token cost, prove it's within budget and fast.

## BATCH 2 — BOT FARMING E2E BENCHMARK
- [ ] 2.1 Login→enter map→farm→EXP tick→level-up→job-change full loop proven with timestamps.
- [ ] 2.2 EXP/kill-rate/deaths per hour recorded (outcome proof), vs before-baseline.

## BATCH 3 — FULL COMPLETENESS AUDIT (sidecar src)
Scan for deadcode/dormant/unconsumed/unwired/incomplete across ALL subsystems; dig deeper where it looks dead (may be incomplete-needed impl). Wire everything.

- [ ] 3.1 conscious (LLM PDCA, triggers, gear advisory, quest, survival strategy, job-change emitters)
- [ ] 3.2 subconscious (DQN `. _train_from_replay`, labeling pipeline, behavior override, reward)  ← verify real training_steps, not code
- [ ] 3.3 reflex (hardwired safety floor, death-loop, fly_wing_escape, survival_reflex)
- [ ] 3.4 economy/gear/progression (gear_progression_planner, buyable_items, potion selection, sell junk)
- [ ] 3.5 routing/pathfinding/map knowledge (portal graph, .dist distmaps, route-calc, spawn handling)
- [ ] 3.6 fleet/self-heal/supervisor (ghost cleanup, self-heal signals, health, party coordination)
- [ ] 3.7 API/bridge/contract (aiSidecarBridge, actor snapshot, command allowlist, ingest v1/v2)
- [ ] 3.8 server_solutions DB-backed store (agnostic facts, seed paths, consumers wired)
- [ ] 3.9 ML/memory (openmemory.db, sidecar_experience, knowledge graph, drift remediation)
- [ ] 3.10 config/deploy (start.sh, fleet_supervisor, launch, systemd/process mgmt, .env)

## BATCH 4 — CROSS-CUTTING + VERIFY-AND-MARK
- [ ] 4.1 Adversarial sweep: "any more missing/flaw/race/loophole/blindspot?" until nothing found.
- [ ] 4.2 Zero stub/todo/pass/dormant grep pass across *.py and bridge *.pl.
- [ ] 4.3 Live-verify every fix (logs/DB rows), commit after each batch, update this checklist.
- [ ] 4.4 Final benchmark-proven production-ready gate.

STATUS LEGEND: [ ] not started · [~] in progress · [x] done · [!] blocked · [P] proven (benchmark/live)
