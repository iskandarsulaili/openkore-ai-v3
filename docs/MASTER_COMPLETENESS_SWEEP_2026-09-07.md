# MASTER COMPLETENESS SWEEP — openkore-ai-v3 (2026-09-07)

MANDATE: implement/integrate/fix/wire/execute/verify EVERYTHING to completeness. Zero mock/stub/placeholder/todo/fixme/dormant/incomplete. Reconcile, NEVER trim features. Verify-before-fix. Benchmark-prove outcome (EXP/kill-rate/level), not intent. Commit after each batch. Update this doc after each batch.

## BATCH 0 — LIVE BLOCKER (gameplay gating)
Goal: bot actually farms end-to-end. This is THE gap between theory and outcome.

- [x] 0.1 ROOT-CAUSE FOUND+PROVEN (map-login barrier): the every-5s keepalive ping in
      aiSidecarBridge.pl fired during LOGIN, coalescing 0B1C (2B) with the 0x0436
      map-login (23B) into a 25-byte segment; map-server (clif_shuffle.hpp expects
      exactly 23) rejected every attempt -> 3.5-day map-login rejection loop.
      FIX: gate keepalive on $mapLoginAcked (in-game only) + import $mapLoginAcked
      into the bridge Globals. PROVEN: bot entered map (02EB Enter Map + combat
      08C8) at 12:48 after restart. First successful map entry in days.
- [x] 0.1b REVERTED a misdiagnosis: the 23-byte pack branch 'v V V V V V C' was
      CORRECT (I miscounted 6 longs); reverted my false fix immediately.
- [~] 0.2 SECOND BUG (now exposed, root-causing): in-game session drops ~40-70s
      after map entry. Evidence: after successful entry (12:48:28, Enter Map +
      combat 08C8), the bot main loop went SILENT for ~43s (no CZ_SYNC 0x0360, no
      0B1C, no actor sends from 12:48:56 to the manual Exit 018A at 12:49:39) ->
      server idle-drops -> "Timeout on Map Server". Root-cause in progress:
      profile sets aiSidecar_ioTimeoutMs 30000 — a slow sidecar POST blocks the
      single-threaded OpenKore main loop (incl. the 12s CZ_SYNC keepalive) up to
      30s -> idle drop. Also: "macro reflex_teleport_escape not found or error in
      queue" (lethal-escape reflex macro undefined in profile) + sidecar emitted
      reflex-lethal_escape_teleport while HP 11%. Fix candidates: (a) lower
      ioTimeoutMs so no single HTTP blocks a whole keepalive window, (b) define
      the reflex macro or make the escape execution non-blocking, (c) verify the
      in-game keepalive fires during long sidecar polls.
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
