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
P1.3 [ ] `_run_one_cycle` is ~3300-line single function, all domains inline; any one
     sub-domain exception fails the whole cycle. FIX: per-domain try/except that
     forwards the error (logs + counts toward that domain's breaker) and continues
     the remaining domains for that bot. (The per-bot loop at 5838 already wraps each
     domain emitter in try/except; audit remaining unwrapped sites.)
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

P2.1 [ ] keep-alive only restarts dead PROCESSES (lifecycle). No "alive but not
     progressing" detection. FIX: stuck-state detector — bot registered+safe but no
     exp/kill/level/position delta for N min → emit recovery (return-to-safe-town →
     re-approach farm) rather than reboot.
P2.2 [ ] Outcome confirmation: acks currently mean "dispatched", not "effect happened"
     (aiSidecarBridge.pl:3605-3652 acks at Commands::run return). Blanked-noop
     commands ack success=1 (3280-3322 cooldown/3349 cast-blank). FIX: track
     intended-effect (exp ticked / moved / hp changed in snapshot window);
     no-effect → treat as failure and re-decide. Sidecar must not clear the intent
     on a blanked-noop ack.
P2.3 [ ] Bridge ack_queue is FIFO, head-only retry, stale head >5s dropped
     (4467-4497). Lost ack → action expires 30s → idempotency cleared → re-emit or
     intent-loss. FIX: ack retry that does not silently drop; sidecar handles
     expired-but-uncertain.

## PHASE 3 — EXACTLY-ONCE ACROSS RESTARTS

P3.1 [ ] Idempotency index is in-memory only (action_queue.py:42,82); rebuilt from
     restored queue only (359). Sidecar+bridge restart → double-exec of a command
     already done. FIX: persist idempotency index alongside queue; bridge-side
     permanent (persisted) command dedup key.
P3.2 [ ] rehydrate() converts dispatched→queued (381) → re-run. With P3.1 the
     already-executed set is excluded.

## PHASE 4 — PROGRESS-OR-ESCALATE (terminating)

P4.1 [ ] Per-bot per-objective cycle ceiling: no progress → escalate to
     intentional(LLM) → still stuck → safe farm reset. Caped so it terminates.
P4.2 [ ] "Cannot calculate a route" / missing .dist field recurring stuck class →
     map-corridor fallback from map DB (a few guaranteed-safe RAW farm maps
     baseline) → resolve to safe fallback, not infinite retry.

## PHASE 5 — EARNING FLOOR (the zero-intervention requirement)

P5.1 [ ] LLM/conscious tier failure must NOT "fail-open to native AI"
     (aiSidecarBridge.pl:2089 / fail-open retained). FIX: fall back to a conservative
     verified-earning routine (stay on farm, drink, keep killing) — a fixed floor
     that provably yields EXP, so a 24/7 run never idles silently.
P5.2 [ ] Fleet-level LLM budget gate so N bots can't burn a provider + degrade
     together.

## PHASE 6 — END-GAME TIER (only on the Phase-5 floor)

P6.1 [ ] Opportunistic/adaptive endgame (economy swings, MVP/WoE timing) via conscious
     tier, STRICTLY on top of the Phase-5 floor. Smart-tier error → keep farming.
P6.2 [ ] RAW-over-internet reconnect/cannot-route bounded by map-corridor fallback.

---

## REGRESSION / VERIFICATION (run per batch)

- [ ] `python3 -m pytest AI_sidecar/tests -x -q` (sidecar suite) green
- [ ] `perl -c plugins/aiSidecarBridge/aiSidecarBridge.pl` clean
- [ ] `python3 -c "import ai_sidecar.app"` imports clean
- [ ] live: start sidecar → /health shows PDCA running + breaker per-bot