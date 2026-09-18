# ROOT CAUSE — sell never fires / zeny 0 (proven 2026-09-18)

## Symptom
Bot reaches Tool Dealer (prt_in 126,76), logs `AI: sellAuto NPC`, then
"The NPC did not respond." → `Auto-sell sequence completed.` → no 0x00C9 → **zeny stays 0** → 500z job-change gate closed.

## Proof
- Outbound packet histogram: `0x0090` (CZ_CONTACTNPC / npc_talk) sent **0 times**; `0x0146` (talk cancel) sent **34–35 times**.
- `'Sent talk: '` = 0 ; `'Initiating the talk (sendTalk)'` = 0 ; `'Sent talk cancel'` = 35.
- `0x00C4` (ZC_SELECT_DEALTYPE = shop dialog open) received **0 times**.
- Main-loop cadence: **10.00 s/iteration** (measured live; should be ~10 ms → ~1000x starvation). Sync every 15–20 s.
- Sidecar `/v1/actions/next` measured **0.61–2.01 s** per call; `actions_next_latency_budget_exceeded` **3541/3541** of all latency breaches.
- Sidecar `/v2/ingest/event` measured **4.945 s**.
- Sidecar env: `OPENKORE_AI_LATENCY_BUDGET_MS=500` (budget breached on every poll).

## Defect 1 (primary) — TalkNPC timeout preempts sendTalk
`src/Task/TalkNPC.pm`: the no-response timeout branch (~line 430) is evaluated
**BEFORE** the step-dispatch branch (~line 436) that consumes the `x` step and
calls `sendTalk`. With a starved main loop the 5 s `npcTimeResponse` always fires
first, so `sendTalk` is never dispatched and EVERY shop dialog aborts.

## Defect 2 (loop starvation) — bridge blocks the main loop on HTTP
`plugins/aiSidecarBridge/aiSidecarBridge.pl` runs blocking socket HTTP inside
`on_mainLoop_post` per iteration:
- `_flush_event_queue()` → POST `/v2/ingest/event` (4.9 s), gated at
  `aiSidecar_eventIngestIntervalMs` = **500 ms** → gate is meaningless.
- `_poll_next_action()` → POST `/v1/actions/next` (0.6–2.0 s).
Combined ≈ 7.5 s/iteration, which is what starves every core timeout.

## Defect 3 (queue flood) — duplicate sellAuto in @ai_seq
`AI::queue()` unshifts with no dedupe; the bridge re-issues `autosell` each poll →
`AI: sellAuto sellAuto sellAuto sellAuto | 4`. Each entry re-drives the talk path.

## Fix order
1. Core `TalkNPC.pm` — do not time out before the talk has been sent.
2. Core `Commands.pm` `cmdAutoSell` — dedupe via native `AI::inQueue('sellAuto')`.
3. Bridge — bound/space the blocking ingest + action polls so the loop keeps cadence.
