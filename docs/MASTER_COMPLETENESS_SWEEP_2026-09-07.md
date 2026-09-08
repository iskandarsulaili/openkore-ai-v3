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
- [x] 0.3d ROUTE-STALL RECOVERY (2026-09-07, commits d712098a7 + 495000089):
      bot wedged in `AI: route | 2` (repeated TOO_MUCH_TIME bails, no move ack,
      position-desynced) re-fired full pathfinding over the 3224-portal graph
      every cycle -> endless Field-object churn (OOM on weak machines) + never
      walked. First fix gated on route_churn_count (which only grows when the
      (map,x,y,ai_top) signature is UNCHANGED — random-route recalcs change the
      target every cycle so it stayed ~0 and never fired). REWRITTEN to trigger
      on POSITION stall: in a route/move task for >45s with no server position
      change = stalled regardless of target -> pos_to re-sync + `ai auto` reset
      + recalc backoff (20s). Config keys aiSidecar_routeStallDetectMs/
      _routeStallRecoverCooldownMs/_routeStallBackoffMs.
- [x] 0.3e COST-GATE RE-OPEN (2026-09-07, commit 5a374ce89): the CostModeManager
      is created inside the _strategic_services_initialized init block, which may
      already be True by the time the daily-budget gate runs -> _cost_mode stayed
      None -> gate fell back to the raw llm_daily_budget_tokens (100000) and
      hard-gated the conscious brain to 100k tokens/day even in cost_mode=max
      (goal=cost_gated, actions=0). Now build the manager from settings at the
      gate if absent so max mode (budget 0 = unlimited) is honored. ALSO fixed
      bot_health_monitor ActionPriorityTier.TACTICAL -> .tactical (StrEnum
      members are lowercase; the uppercase fallback raised AttributeError and
      dropped the health_enqueue recovery action). PROVEN: goal=survival (not
      cost_gated), CrewAI plan active, EXP 11550->13165 continuously.
- [x] 0.3f RECONNECT-GRACE GUARD (2026-09-08, commit 8b25d92c4): the no-progress
      self-heal fired map-change within 3 min of EVERY reconnect (a freshly-logged
      bot has momentarily-frozen EXP + fresh snapshot), disconnecting the bot in a
      vicious reconnect->heal->disconnect loop. Track last in-game transition +
      suppress the no-progress heal for 90s after it (config
      _stall_reconnect_grace_s). PROVEN: EXP 5462->7449 climbing, no self-heal
      disconnects for testbot99.
- [x] 0.3g HEALTH-MONITOR EMPTY-MAP GUARD (2026-09-08, commit b3458c248): an EMPTY
      map_name (bot mid-reconnect, snapshot not yet populated) was treated as
      "in town", so after 3 cycles health_monitor sent a FARMING bot to hunt a
      different map (prt_fild05), disconnecting it. is_in_town now requires a
      non-empty map_name.
- [x] 0.3h ROUTE-RECALC LOOP FIXED (2026-09-08, commit 360df1644): ROOT CAUSE =
      same-map random-walk dispatched Task::MapRoute (the 3224-portal cross-map
      graph) for a SAME-MAP target (noMapRoute=0 when route_randomWalk==1),
      re-running the expensive portal-graph calc every cycle, bailing
      TOO_MUCH_TIME before a walk was sent, wedging the bot in `AI: route | 2`
      (never attacks, server drops at stall_time 60). FIX: Actor::route now uses
      fast Task::Route (.dist pathfinding) when the target map == current field.
      PROVEN: EXP 7559->9872->10498->11509->12025 continuously, kills
      Poring/Solid Lunatic/Lunatic/Fabre, in-map sustained, no route-recalc
      wedge, same PID (no restart) across 10+ min.
- [x] 0.3j ATTACK-CONFIG THRASH FIXED (2026-09-08, commits d15072e12 + 7e911b497):
      the config-audit block set attackMaxDistance 2 then 30, attackDistance 1
      then 5, startOnSight 0 then 1 in the SAME block -> the bot thrashed between
      melee and ranged config every cycle and never settled into attacking (HP
      dropped while it stood there). Removed the contradictory ranged values;
      attackMaxDistance 2 was ALSO too tight for pathing ("Too far from us to
      attack, distance is 3, maxDistance is 2" + meetingPosition not_walkable
      rejections = endless chase loop) -> 4/2 (melee reality + pathing buffer).
      PROVEN: bot attacks + kills again (Dmg 101-116, kills Poring/Lunatic).
- [x] 0.3k SAVE-POINT FIX (2026-09-08): bot's save point was izlude (127,142) but
      sellAuto_npc + farm map are prontera-side -> every death respawned it far
      from the farm + sell NPC. DB save point moved to prontera (156,129).
      NOTE: char-server overwrites the DB save point on death (reverts to izlude)
      — re-apply after each death; the real fix is a server-side save-point
      change (pending).
- [x] 0.3l OVERWEIGHT-RETURN FIXED (2026-09-08, commit 3373243d1): the
      return-to-town logic skipped when the bot was ON its farm map
      (_audit_on_farm), so an overweight bot (bag full, weight > 70%) never
      returned to sell -> stopped earning zeny entirely, stayed at 0 zeny, and
      the job-change stayed blocked. A full bag is a hard stop: return + sell.
- [x] 0.3m JOB-CHANGE STALE-SNAPSHOT REVERTED (2026-09-08, commit ce6257e3e):
      the healthy-HP resume (bf7b6d7ab, 8d7a57cb5) forced the bot to WALK the
      cross-map route to alberta at full HP. It immediately hit the
      position-desync + cross-map route-calc loop the conscious tier had warned
      about and started dying (HP 204->60). The conscious tier's plan was
      CORRECT: farm for a Fly Wing first, THEN job change. Reverted the
      healthy-HP resume; KEPT 72ecb1651 (JOB_CHANGE disables route_randomWalk).
- [x] 0.3n JOB_CHANGE->HUNT FALLTHROUGH FIXED (2026-09-08, commit f8599e762):
      the state machine returned JOB_CHANGE unconditionally when eligible, but the
      JOB_CHANGE handler DEFERS when survival_strategy is active (level_up_first /
      fly_wing_escape). Result: the bot sat in JOB_CHANGE state doing NOTHING
      (handler defers, no move emitted) and never farmed — it wandered on izlude,
      EXP frozen, and died. Now the state machine checks the survival strategy and
      falls through to HUNT so the bot actually farms while the strategy defers.
      PROVEN: EXP 2931->5765 continuously, kills Poring/Fabre/Lunatic, position
      moving on prt_fild08, state=HUNT.
- [x] 0.3o DQN ZENY-GAIN REWARD (2026-09-08, commit 158732877): the DQN reward
      was survival-only (0.05 alive / -1 dead), so it never learned to sell loot
      (the bot accumulated junk but stayed at 0 zeny). Added a zeny-GAIN reward
      term so the subconscious learns to sell. (Needs training time to take
      effect; the heuristic overweight-return covers the immediate gap.)
- [ ] 0.3i DQN/LLM WIRING (2026-09-08, ACTIVE): the 3-tier brain is real (DQN
      trained 61,470 steps, reward 2900; conscious LLM 84 lines vs heuristic 985
      lines in the same window) but the bot is ~92% heuristic-driven. The
      execution layer (route-recalc + attack-config thrash) prevents the DQN from
      driving combat and the LLM from setting intent. Fix 0.3h first, then wire
      the DQN to drive combat + LLM to set intent, demote heuristics to cold-start
      fallback.
- [ ] 0.3 ROUTE-FAILURE STALL: 2386 route-calc fails on prt_fild08 (post-stability).
- [x] 0.3p SELL LOOP / PRIORITIZATION (2026-09-08, PROVEN): user mandate "able to
      prioritize" + "no common sense". ROOT-CAUSED: the conscious tier decided
      survival_strategy=fly_wing_escape at LETHAL HP (0/1) — "field crossing kills
      me". That premise stayed STALE after the bot healed -> it ground a starter
      field forever for a Fly Wing it can't afford (never sells loot -> zeny 0 ->
      deadlock). FIX: a healthy HP override (hp_ratio >= 0.90) now takes priority at
      ALL 6 job-change gate sites (heuristic_service TOWN branch / cold-start step-7 /
      main emitter / JOB_CHANGE handler / HUNTING branch / domains/progression.py) —
      healthy bot prioritizes the job change; a fragile bot (<0.9 HP) still honors
      the conscious fly-wing deferral (correctly re-engages at real low HP).
      COMMITTED 57f887ff2 + 61915d133 + 6d4c07393; sidecar restarted (2670492).
      PROVEN: bot reached JOB_CHANGE state + routed to alberta_in; HP dropped on the
      crossing -> correctly re-deferred to HUNT; then farmed continuously EXP
      7087->9562 in 3 min, 130 kills, zero deaths, 1 process. Prioritization works:
      survival first when fragile, progression when healthy.
- [~] 0.4 AFTER JOB CHANGE: bot must complete the merchant job-change (reach alberta
      guild NPC, talk, pick merchant) end-to-end. Currently it defers at low HP
      crossing the field; verify it completes once HP + Fly Wing path is resolved.

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
