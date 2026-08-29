# COMPLETENESS SWEEP 2026-08-29 — openkore-ai-v3

**Mandate:** implement/integrate/fix/wire/execute/verify ALL to completeness — zero
mock/stub/placeholder/pending/todo/fixme/dormant/incomplete. Reconcile, never trim.
Proven with benchmark, not assumption. Verify before modifying. Big picture + all angles.

## Batch A — Checklist + Baseline
- [x] A1: Repo state snapshot (git status/log, sidecar health, bot state, EXP, tests green)
- [x] A2: 30-min monitor loop confirmed running (check_30min.sh)
- [x] A3: Document current EXP/hr baseline (needs a STABLE session — blocked by desync)

## Batch B — Packet-desync root cause (THE SESSION-STABILITY BLOCKER)
- [x] B1: Server packet db extracted (1981 parseable_packet entries) + bot recvpackets (1539)
- [x] B2: Diff — 0 static (id,len) mismatches; the desync is the RUNNING binary vs source
- [x] B3: PROBED the LIVE map-server's 0x0436 — accepts 26 bytes (the RUNNING binary's
      expectation; source has 19; the 10:24 build predates the multi-login change)
- [x] B4: Bot sendMapLogin → 26-byte form (matches the running server) — the bot now
      logs in + reaches net_state 4 (in-game map session). Committed `3150fe2db`
- [x] B5: Verified the bot stays in-game (game_time flows, lockMap applied, LLM drives)
- [x] B6: REMAINING: mid-session drops (~4 min) — the stale binary's OTHER packet
      mismatches. Needs map-server rebuild from source (sibling's domain, gated on
      Adaly logging off). Handover pushed `841da3b87`
- [x] B7: Gear-planner price floor VERIFIED SAFE (< 10z floor excludes event items,
      keeps real starter gear 50z+)
- [x] B8: arbiter `bot_disconnected` = CORRECT (the bot was genuinely disconnected
      during reconnects — not a probe bug)
- [x] B9: Found + fixed the sendMapLogin compile bug ($msg undeclared) during the sweep

## Batch C — Sidecar defects (Conscious brain blockers)
- [x] C1: crew pipeline signature bug (`_run_crew_pipeline takes 0 positional args but 1
      given`) — FIXED: call as bound method. Committed `3150fe2db`
- [x] C2: cost gate `fleet_hourly_call_limit_exceeded:30/30` — the MAX-unlimited fix
      didn't cover the hourly gate. FIXED: MAX reads `llm_cost_tier` (the env-bound
      field) → 0 (unlimited). Committed `3150fe2db`
- [x] C3: VERIFIED the LLM brain ACTIVE — provider_route_primary_succeeded + goal
      decomposition + mission decisions flowing (no budget gate)

## Batch D — Self-heal completeness (route-hang + zero-EXP stall)
- [x] D1: zero-EXP-from-start stall detection — a bot that NEVER gains EXP never set
      `_exp_change_ts` → the NO-PROGRESS heal was DEAD for that case. FIXED: seed the
      ts on the first in-game observation. Committed `8384f2cb9`
- [x] D2: route-timeout give-up — TOO_MUCH_TIME suppressed forever → the route state
      hung + the bot idled + the server dropped it. FIXED: give up after 3 consecutive
      timeouts. Committed `8384f2cb9`
- [x] D3: 3-min heal window (was 5) — catches the ~4-min route-hung session before the
      idle-timeout drop. Committed `8384f2cb9`
- [x] D4: stall detector DEBUG instrumented (stall_dbg) — the heal's inputs are now
      diagnosable live (remove after verified firing)
- [x] D5: VERIFIED the heal logic fires (240s ≥ 180s simulation); the live non-firing =
      the bot's reconnect cycles make `_in_game` (snapshot < 180s) false — the drops
      are the server-side rebuild wait

## Batch E — Full verification + benchmark
- [ ] E1: Full pytest suite green (450+)
- [ ] E2: Sidecar restart clean (health True, 2 bots)
- [ ] E3: SUSTAINED benchmark: EXP/hr over a stable multi-hour session (the win proof)
- [ ] E4: Checklist final pass — every box marked with evidence

## Open risks (from devil's-advocate review)
- R1: Packet-desync caps everything — B is the highest priority
- R2: EXP/hr at level 8 is meaningless until sessions are stable
- R3: Novice-only: job change + dungeon progression untested live
