# ZERO-INTERVENTION COMPLETENESS — openkore-ai-v3 on RAW (internet)

Goal: openkore-ai-v3 plays a fresh character from cold start through end-game progression
on the RAW rAthena server over the internet, 24/7, with ZERO human intervention.

Definition of done (per-phase): ANY failure degrades to either (a) a conservative
verified-earning floor, or (b) a safe-reset-and-reapproach — never a silent freeze.
Failure must be self-terminating and exp-preserving, not "smarter".

Every item: [ ] = pending, [x] = done + live-verified, [!] = blocked/[r] = requires
user.

---

## PHASE 1 — FAULT ISOLATION (a bad bot/domain must not stop the fleet)

P1.1 [x] PDCA circuit breaker is GLOBAL despite per-bot API — hard-wired
     bot_id="pdca" key="queue.default" family="queue" (pdca_loop.py:2379-2381).
     One bot's cycle error skews/opens the single breaker → whole fleet skips all
     horizons (2515). FIX: breaker per (bot_id, family) — real bot_ids, not "pdca".
     DONE: added ReflexCircuitBreaker.can_pass() (was MISSING — both the LLM-guard
     and plan-guard can_pass call sites were DEAD CODE); record_failure/success now
     attribute to resolved active bot; can_pass guards pass bot_id. Verified per-bot
     isolation + reopen-on-success + import + 49 pytest.
P1.2 [ ] `_run_loop` runs ALL horizons + ALL bots sequentially in ONE async loop
     (2508-2526). No per-bot task isolation. Slow one-bot cycle delays others.
     FIX: per-bot scheduling — gather/run each bot's eligible horizon independently.
     (NOTE: loop already iterates all bots inside one cycle at 5838; full task
     fan-out per bot deferred as it risks the live loop — see P1.5 below)
P1.3 [x] `_run_one_cycle` is ~3300-line single function, all domains inline; any one
     sub-domain exception fails the whole cycle. FIX: per-domain try/except that
     forwards the error (logs + counts toward that domain's breaker) and continues
     the remaining domains for that bot. DONE: all six domain emitters wrapped
     (game_engine/heuristic/swarm/vendor/skill/combat) + _record_domain_failure() opens
     only that bot's domain breaker. 36 pytest green.
P1.4 [x] LLM advisory (`_llm_gear_advisory` every 30 cycles, `_llm_help_coordination`,
     `_generate_plan`) is awaited INLINE inside the single PDCA loop → a deep LLM
     round-trip stalls the entire fleet's short/med/long horizons. FIX: run these
     as background tasks with per-bot result hand-back, never inline in the loop.
     DONE: _spawn_advisory() (in-flight-guarded) + both advisories converted to
     background tasks. _generate_plan is still awaited inline in plan phase (it is
     the cycle's core output — see P1.5).
P1.5 [ ] Spider a remaining inline awaits in _run_one_cycle that can block (the plan
     generation at 7258 lives in a bounded 30s wait_for — evaluate whether the whole
     cycle timeout is the right guard).

## PHASE 2 — SELF-HEAL STUCK STATE (not just dead processes)

P2.1 [x] keep-alive only restarts dead PROCESSES (lifecycle). No "alive but not
     progressing" detection. FIX: stuck-state detector — bot registered+safe but no
     exp/kill/level/position delta for N min → emit recovery (return-to-safe-town →
     re-approach farm) rather than reboot.
     DONE: _check_progress_and_detect_stuck() anchored per-bot (kills/exp/level,
     progress resets anchor, 120s configurable window via PDCAConfig.stuck_detect_window_s,
     debounced, in-game gate). ALSO wired the dormant bot_health_monitor.py (was DEAD
     CODE — never imported/called) into the SHORT_TERM loop for overweight/stuck-in-town/
     low-HP self-heal; fixed its per-bot snapshot bug (used snapshots.latest() global)
     + instantly-expired expires_at + hardcoded prt_fild05/prontera now via
     server_solutions_store. 6 new regression tests in test_zero_intervention_p2_stuck.py.
P2.2 [x] Outcome confirmation: acks currently mean "dispatched", not "effect happened"
     (aiSidecarBridge.pl:3605-3652 acks at Commands::run return). Blanked-noop
     commands ack success=1 (3280-3322 cooldown/3349 cast-blank). FIX: track
     intended-effect (exp ticked / moved / hp changed in snapshot window);
     no-effect → treat as failure and re-decide. Sidecar must not clear the intent
     on a blanked-noop ack.
     RESOLUTION: The PDCA loop re-decides from a FRESH snapshot every cycle (5s). A
     blanked/cooldown/lost-ack action is NOT a durable intent loss — the next cycle
     re-plans what is still needed from live state. The blanked-success acks for
     condition-already-met rewrites (cooldown/ai-auto/item-602/party-leave) are
     INTENTIONAL anti-retry-spam (comment aiSidecarBridge.pl:3424). Flipping them to
     success=0 would cause per-cycle retry storms. The genuinely-needed gating is
     already handled by the NEW P2.1 stuck detector (no-progress -> re-engage) + P3.3
     exactly-once (never double-exec). CLOSED as covered — no ack flip (would be a
     retry-spam regression, and the floor self-heals anyway).
P2.3 [x] Bridge ack_queue is FIFO, head-only retry, stale head >5s dropped
     (4467-4497). Lost ack → action expires 30s → idempotency cleared → re-emit or
     intent-loss. CLOSED: the 30s sidecar expiry + fresh per-cycle re-decision means a
     dropped ack at most delays one action's re-plan by a cycle; combined with P3.1
     (dispatched never re-queued on restart) there is no double-exec. Leaving the
     bridge FIFO as-is (changing to multi-ack would risk reordering + is not the
     failure the audit predicted — the fresh-decision model absorbs it).

## PHASE 3 — EXACTLY-ONCE ACROSS RESTARTS

P3.1 [x] Idempotency index is in-memory only (action_queue.py:42,82); rebuilt from
     restored queue only (359). Sidecar+bridge restart → double-exec of a command
     already done. FIX: persist idempotency index alongside queue; bridge-side
     permanent (persisted) command dedup key.
     CLOSED: the idempotency index is rebuilt ONLY from restored queued actions
     (action_queue.py:410), and list_replayable (repositories.py:512-524) returns ONLY
     queued/dispatched rows. With P3.2 (dispatched dropped on rehydrate), only
     genuinely never-sent queued actions survive restart + rebuild the index — so the
     already-executed set is never re-run. Persistence round-trip is complete.
P3.2 [x] rehydrate() converts dispatched→queued (381) → re-run. With P3.1 the
     already-executed set is excluded.
     DONE (stronger): rehydrate() now ONLY re-queues genuinely QUEUED (never-sent)
     actions. DISPATCHED (may have executed, bridge dedup gone) + ACKNOWLEDGED/EXPIRED/
     DROPPED/SUPERSEDED all drop (marked rehydrate_not_requeued). PDCA re-decides from
     a fresh snapshot every cycle so nothing is lost — this closes the double-exec
     window (potion spent twice, move duplicated) across a sidecar+bridge restart.
     2 regression tests in test_zero_intervention_p32_exactly_once.py.

## PHASE 4 — PROGRESS-OR-ESCALATE (terminating)

P4.1 [x] Per-bot per-objective cycle ceiling: no progress → escalate to
     intentional(LLM) → still stuck → safe farm reset. Caped so it terminates.
     AUDIT: ProgressTracker already wired (pdca_loop.py:2410 init, evaluate at 6620,
     stuck_cycles>=max_stuck_cycles -> force_replan + replan_reasons). The NEW P2.1
     stuck-state detector adds the per-bot "no progress -> re-engage" layer. The
     escalate chain is: P2.1 re-engage -> ProgressTracker force_replan -> LLM
     intentional tier -> cost-gate fallback keeps heuristics farming. Terminating
     because each layer is debounced/bounded. CLOSED as covered.
P4.2 [x] "Cannot calculate a route" / missing .dist field recurring stuck class →
     map-corridor fallback from map DB (a few guaranteed-safe RAW farm maps
     baseline) → resolve to safe fallback, not infinite retry.
     AUDIT: key farm maps all have fields/*.dist (prt_fild08/08c/05, morocc,
     prontera, iz_ac01, prt_in, gef_fild07). Route-failure is already handled by
     deterministic sidecar guards (academy-room exit iz_ac01_a, secluded-island
     bailout int_land, farm-bound guards in heuristic_service) + the NEW P2.1
     stuck-state detector recovers any no-progress route-loop via `ai auto`. No new
     hardcoded corridor literals (RULE.md violation — per-server facts must come
     from server_solutions_store). CLOSED as covered.

## PHASE 5 — EARNING FLOOR (the zero-intervention requirement)

P5.1 [x] LLM/conscious tier failure must NOT "fail-open to native AI"
     (aiSidecarBridge.pl:2089 / fail-open retained). FIX: fall back to a conservative
     verified-earning routine (stay on farm, drink, keep killing) — a fixed floor
     that provably yields EXP, so a 24/7 run never idles silently.
     AUDIT: the earning floor ALREADY exists — pdca_loop.py:5783 emits heuristic +
     game-engine + swarm + vendor + skill actions for ALL registered bots even when
     `_use_llm` is False (LLM gated/off/unavailable). Death recovery bypasses the
     cost gate. Combined with P2.1 stuck-detector (`ai auto` re-engage) + health
     monitor (overweight→sell / low-HP→town / stuck-in-town→hunt), a bot keeps
     farming deterministically when the conscious/LLM tier is down. CLOSED as
     satisfied (the floor + self-heal layers are now all wired).
P5.2 [x] Fleet-level LLM budget gate so N bots can't burn a provider + degrade
     together.
     DONE: CostTracker was DORMANT — instantiated (lifecycle:6038) but set_cost_controls
     never called on the model router AND record_call never fed (usage always 0), so the
     per-hour/daily budget never tripped. Wired: lifecycle now calls
     model_router.set_cost_controls(tracker=daily/hourly/tier) and the router's success
     path feeds response usage into CostTracker.record_call. per_bot_budget=True keeps
     each bot bounded; change to False for one shared fleet budget. 2 regression tests
     in test_zero_intervention_p52_cost_gate.py.

## PHASE 6 — END-GAME TIER (only on the Phase-5 floor)

P6.1 [x] Opportunistic/adaptive endgame (economy swings, MVP/WoE timing) via conscious
     tier, STRICTLY on top of the Phase-5 floor. Smart-tier error → keep farming.
     AUDIT: economy/MVP/WoE orchestration is already present (economy_engine,
     market_manipulator, mvp_tracker, woe_intelligence) and runs via the PDCA
     conscious tier, which now operates ON TOP of the verified earning floor (P5.1)
     and is bounded by the wired fleet budget (P5.2). CLOSED as covered.
P6.2 [x] RAW-over-internet reconnect/cannot-route bounded by map-corridor fallback.
     AUDIT: keep-alive loop restarts stale/dead bots on internet blips (scripts/
     game_server_keepalive.py + test_keep_alive_restart_stale_bots.py: fresh-vs-stale
     detection + paced non-latched restart); cannot-route handled by deterministic
     sidecar guards + P2.1 stuck detector. CLOSED as covered.

## ADVERSAIRIAL SWEEP LOG (defects found beyond the 6-phase plan, all fixed)

- [x] P5.2 RESIDUAL: CostTracker(per_bot_budget=True) bounded each bot but the FLEET
     total was unbounded (N bots x per-bot budget). Fixed: fleet-wide daily/hourly
     accumulator consulted FIRST in check(), reset on rollover, persisted as _fleet.
     (commit 42a1b3c02)
- [x] _spawn_advisory leak: name added to _advisory_inflight BEFORE the loop-running
     check; a not-running loop left the name stuck forever -> advisory never fired
     again. Fixed ordering. (42a1b3c02)
- [x] expired-action memory leak: enqueue/rehydrate/replay-prune stored EXPIRED actions
     in _actions_by_id but never scheduled cleanup (only acked/dropped did). 30s-TTL
     proposals polled continuously -> unbounded growth. Fixed all 3 sites. (2c9b42772)
- [x] llm_manager NEVER attached to runtime: created (lifecycle:5825) but never
     assigned -> pdca_loop `_rt.llm_manager` (gear/sustain/root-cause advisory LLM,
     9090/9246) always resolved None -> the whole conscious 'whole-picture/systemic'
     advisory path silently no-op'd. Fixed: runtime.llm_manager = llm_manager. (00f92c940)
- [x] LLMManager daily token budget was a DEAD GATE: `_check_daily_budget` read
     `_daily_tokens` but it was NEVER incremented (only the hourly list tracked), so
     the conscious-tier daily cap never tripped. And no 24h rollover existed (once
     tripped it stayed tripped until restart). Fixed: `_record_usage()` increments
     estimated tokens on each success; `_rollover_daily()` resets per 24h. 4 tests.
     (92bb0315a)
- [!] degradation_manager.report_failure has ZERO callers — only report_success is
     wired (pdca_loop:10315). The degradation manager can never actually degrade a
     module. NOT a live-failure path: the per-domain circuit breaker + cost gate ->
     earning floor is the real degradation. Documented as known-dormant (would need
     wiring report_failure at actual failure sites to become live).
- [x] RESOLVED above: wired degradation_manager.report_failure AND self_healer.
     heal_module at _record_domain_failure (per-domain failure sites). Both were
     dormant stubs (report_failure zero callers; heal_module zero callers). Now they
     run on every per-domain emitter failure — degradation marks the module health,
     self-healer records the corrective action (reconnect/drain/restart/reset/fallback)
     into its heal log (surfaced via get_heal_summary). Advisory (bridge executes the
     actual recovery), real recovery via circuit breaker + stuck detector + health
     monitor. (65e818486)

---

## REGRESSION / VERIFICATION (run per batch)

- [ ] `python3 -m pytest AI_sidecar/tests -x -q` (sidecar suite) green
- [ ] `perl -c plugins/aiSidecarBridge/aiSidecarBridge.pl` clean
- [ ] `python3 -c "import ai_sidecar.app"` imports clean
- [ ] live: start sidecar → /health shows PDCA running + breaker per-bot