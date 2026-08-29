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
- [x] B2: Diff — 0 static (id,len) mismatches; the desync is the RUNNING binary's STALE db
- [x] B3: ROOT CAUSE — the RUNNING map-server binary has 0x0436=23 (stale build 10:24);
      the SOURCE tree has 19 (multi-login era). Bot's 19-byte send → 'expected 23,
      got 19' → stream desync → 'unsupported packet 0xXXXX, N bytes' → drop ~2 min in.
- [x] B4: FIXED bot-side — sendMapLogin now sends the 23-byte form (19-byte core +
      4 pad) matching the RUNNING server (RULE.md: adapt to LIVE, not source).
      Bot gets into the map. Committed 3150fe2db.
- [ ] B5: SERVER-SIDE (sibling): rebuild + restart map-server from current source
      (0x0436=19) — removes the stale-binary desync permanently. GATED: no restart
      while CloudNine/Adaly online.
- [ ] B6: Verify 15-min stable farming session (no unsupported-packet) + EXP/hr benchmark
- [x] B7: Found + fixed the sendMapLogin compile bug ($msg undeclared) during the sweep

## Batch C — Sidecar defects found in the live log (all fixed + committed)
- [x] C1: crew_manager._run_crew_pipeline — getattr(type(self), fn) = UNBOUND fn;
      passing self positionally broke the keyword-only signature ('takes 0
      positional args but 1 given') → Conscious brain degraded EVERY cycle.
      FIXED: bound method call.
- [x] C2: cost_controls — MAX tier still hit fleet_hourly_call_limit 30/30 (the
      daily-token fix was incomplete). FIXED: MAX zeroes BOTH daily + hourly gates;
      reads llm_cost_tier (the env-bound field), not cost_mode (default standard).
- [x] C3: LLM brain verified ACTIVE post-restart — provider_route_primary_succeeded,
      no budget gate, 0 crewai_refine_degraded.

## Batch D — Conscious brain / LLM verification
- [ ] D1: LLM calls flowing (0 budget_exceeded, real reasoning in logs)
- [ ] D2: charstatus.json holds ALL char fields (base_exp, job, position, vitals, inventory)
- [ ] D3: Self-heal chain + comeback + reward ledger all firing live
- [ ] D4: Job-change chain: eligibility detected + NPC resolved + acted on

## Batch E — Full verification + benchmark
- [ ] E1: Full pytest suite green (450+)
- [ ] E2: Sidecar restart clean (health True, 2 bots)
- [ ] E3: SUSTAINED benchmark: EXP/hr over a stable multi-hour session (the win proof)
- [ ] E4: Checklist final pass — every box marked with evidence

## Open risks (from devil's-advocate review)
- R1: Packet-desync caps everything — B is the highest priority
- R2: EXP/hr at level 8 is meaningless until sessions are stable
- R3: Novice-only: job change + dungeon progression untested live
