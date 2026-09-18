# RAW AI BOT — FULL-STACK COMPLETENESS CHECKLIST (2026-09-18)

**Objective:** make the AI bot (openkore-ai-v3 + rAthena AI World + launcher/DLL + services)
fully complete and production-ready. Zero mock/stub/placeholder/pending/todo/fixme/dormant/incomplete.
Reconcile — never trim. Prove every item with a benchmark or live evidence, not theory.

**Ultimate measurable goal:** the bot changes job (Novice → first job) and then plays
continuously to end-game, progressing continuously. Current blocker: sell → zeny → 500z gate.

**Baseline (measured 2026-09-18 ~15:00):**
- Bot: testbotA / testbot99 (account 2000011, char_id 90071254), base_level 48, job_level 10, **zeny 0**
- Sidecar: pid 1282436, **uptime 1d04h** → running code is STALE (all session fixes are newer)
- Main-loop cadence: **1 iteration / 20s** (target ~0.1s) — the root blocker
- Bridge blocking calls per iteration: `/v2/ingest/event` **4.4–4.7s**, `/v1/actions/next` **0.5–1.9s**
- Watchdog: circuit breaker previously stranded the bot silently (fixed this session)
- Server: `use_dnsbl: yes` adds blocking DNS lookups inside the login auth path (0.26–5.9s/zone)

---

## BATCH 1 — Unblock the main loop (CRITICAL PATH)

The main loop must never block on HTTP. Every sidecar call made from
`on_mainLoop_post` is synchronous socket I/O; a slow sidecar stalls the whole bot and
starves every core timeout (this is what prevents `sendTalk`/sell/zeny).

- [ ] 1.1 Move per-iteration sidecar HTTP off the main loop (async/queued, non-blocking)
- [ ] 1.2 `_flush_event_queue` → never blocks the loop; bounded + coalesced
- [ ] 1.3 `_poll_next_action` → non-blocking dispatch; results drained on later iterations
- [ ] 1.4 Bound every HTTP call with a hard timeout << loop budget
- [ ] 1.5 Verify: main-loop cadence back to ~0.1s (benchmark before/after)

## BATCH 2 — Sidecar freshness + performance

- [ ] 2.1 Restart sidecar so all session fixes are live (verify via code mtime vs process start)
- [ ] 2.2 Profile `/v1/actions/next` (0.5–1.9s) and `/v2/ingest/event` (4.4–4.7s) hot spots
- [ ] 2.3 SQLite (782MB) persist cost on the request path — offload/batch
- [ ] 2.4 LLM calls must never run inside a poll path
- [ ] 2.5 Verify: p95 latency for both endpoints, with numbers

## BATCH 3 — Sell → zeny → job change (the actual goal)

- [ ] 3.1 Confirm `sendTalk` (0x0090) fires and the shop dialog (0x00C4) opens
- [ ] 3.2 Confirm the sell list (0x00C9) arrives and items are marked sellable
- [ ] 3.3 Confirm a real sale → zeny > 0 (DB-verified)
- [ ] 3.4 Confirm the 500z job-change gate is reachable and the job change completes
- [ ] 3.5 Verify: zeny non-zero in DB + job changed (live proof, not logs alone)

## BATCH 4 — Session stability (no silent drops)

- [ ] 4.1 0x05FC session-killer eliminated (already fixed — re-verify after restart)
- [ ] 4.2 Login/char/map legs stable over the INTERNET path (public tunnels)
- [ ] 4.3 No self-issued Exit (0x018A) from timeouts
- [ ] 4.4 Verify: single continuous in-game session > 30 min, zero disconnects

## BATCH 5 — Server-side login latency (needs approval)

- [ ] 5.1 `use_dnsbl: yes` blocking lookups in the auth path (0.26–5.9s/zone/login)
- [ ] 5.2 Decide: disable, or swap to a responsive zone, or move off the auth path
- [ ] 5.3 Verify: login reply time before/after

## BATCH 6 — Reflex / RULE.md compliance

- [ ] 6.1 Reflex emits only instant combat/safety actions (no strategy)
- [ ] 6.2 Sell-trip latch honored by all reflex paths
- [ ] 6.3 Verify: grepped all reflex emitters against RULE.md lines 37-49

## BATCH 7 — Dead code / dormant / incomplete sweep

- [ ] 7.1 Duplicate `edge/edge_case_handler.py` (181 lines, 0 importers) — reconcile, don't trim
- [ ] 7.2 `sellAuto 0` forced every cycle vs queued sellAuto — reconcile
- [ ] 7.3 Any defined-but-never-called helpers / write-only fields
- [ ] 7.4 Verify: report every finding + disposition

## BATCH 8 — Benchmark + full E2E proof

- [ ] 8.1 Loop cadence benchmark
- [ ] 8.2 Endpoint p95 benchmark
- [ ] 8.3 Full E2E: login → farm → sell → zeny → job change (timestamps + DB rows)
- [ ] 8.4 Continuous-play proof (EXP/level progression over time)

---

## STATUS LOG

| Batch | Item | Status | Evidence |
|-------|------|--------|----------|
| 1 | 1.1–1.5 | PENDING | — |
| 2 | 2.1–2.5 | PENDING | — |
| 3 | 3.1–3.5 | PENDING | — |
| 4 | 4.1–4.4 | PARTIAL | 0x05FC fixed (0 kills since 14:35); internet path proven |
| 5 | 5.1–5.3 | PENDING | DNSBL cost measured (0.26–5.9s) |
| 6 | 6.1–6.3 | PENDING | — |
| 7 | 7.1–7.4 | PENDING | — |
| 8 | 8.1–8.4 | PENDING | — |

## SESSION COMMITS SO FAR

```
0c135c999 fix(net): dual-stack hostname reported as 'couldn't connect (error code 22)' when it HAD connected
448378071 fix(watchdog): circuit breaker had NO recovery path — bot stayed dead, silently
591829e35 fix(net): make the 0x05fc conn-info packet opt-in (its split delivery killed the session)
0e5bec321 fix(net): 0x05fc conn-info must carry a NUMERIC address (server killed the session)
ac9b78206 fix(core): sendTalk was preempted by the no-response timeout; dedupe sellAuto queue
586b19624 fix(sidecar): sellAuto_npc_steps 'c r1 n' -> 's'
```
