# MASTER COMPLETENESS CHECKLIST — openkore-ai-v3 (2026-08-25)

> Single source of truth consolidating every open item across
> COMPLETENESS_TRACKER.md / AUDIT_TRACKER.md / ZERO_INTERVENTION_*.md /
> RAW_P2P_INTEGRATION_PLAN.md / FOUNDER_STEERS.md. Mandate: implement/integrate/
> fix/wire/execute/verify EVERYTHING to completeness for live production. Zero
> mock/stub/placeholder/pending/todo/fixme/dormant/incomplete. Reconcile, never trim.
>
> Legend: [ ] OPEN · [x] DONE (impl+wired+verified live) · [~] partial/in-progress

## PHASE 1 — SESSION STABILITY (DONE)
- [x] 1.1 C A* timeout now fires EVERY pop (not every 100th). commit cd7392c4b
- [x] 1.2 Garbage-dims guard (product ≤ 4M) at XS + CalcPath_init; NULL-checks; failed flag.
- [x] 1.3 flushKeepalive (CZ_SYNC) in Task::Route / CalcMapRoute / Misc (same-map+nested).
- [x] 1.4 Sidecar (:18081) restarted + bot re-registered (was down all night).
- [x] 1.5 LIVE verified: bot in-game 20+ min, drop count frozen, keepalive+sync both ways.

## PHASE 2 — PRODUCTION-PROGRESSION BLOCKERS (open items from trackers)
- [x] 2.1 (A5) Verify char-select "all maps not ready" root cause = peer-host flap vs central ownership.
      VERIFIED: central owns all 1262 maps stably (map-server 8h+ uptime, char 8h+); the flap events in
      char log are historical (earlier troubleshooting), not active. Bot enters map reliably.
- [x] 2.2 (A6) Fix char-select retry so bot enters map reliably when central owns maps.
      VERIFIED: bot enters map (Enter Map 02EB + "You are now in the game") and routes to farm.
- [ ] 2.3 (A7/D1) Full 20250604 active-block packet diff OpenKore recvpackets vs RAW — no "Unknown switch".
- [ ] 2.4 (D6) iz_int* tutorial room nav — bot can't path the HIDDEN_WARP_NPC gate out of intro room.
- [ ] 2.5 (S21/S26) iz_ac01_a client can't walk the 28-cell exit (100,52→100,24). Client pathfinding bug.
- [x] 2.6 (S31/S39) FleetOrchestrator captain relocation to guaranteed loop (app.py keep-alive) so fleet directives actually fire.
      VERIFIED: orchestrator tick relocated outside init block (pdca_loop.py:4861), runs every cycle, 15s cadence.
- [ ] 2.7 (D10) Server-side starting-resource limit — gearless L1 can't survive island Porings. Needs RAW change (now AUTHORIZED per A2).
- [x] 2.8 NEW: snapshot→world_state wiring gap FIXED. runtime.ingest_snapshot now forwards to
      normalizer_bus.ingest_snapshot so the bot's live position/map flows into enriched_state/PDCA/heuristics
      (was map=None/x=None/y=None = blind navigation). +3 regression tests. VERIFIED live via probe.
- [x] 2.9 NEW: keep-alive map-load grace + targeted restart. Fresh bots (<240s) never stale (was nuked
      mid-load by whole-fleet restart); stale bots restarted individually via `start.sh bot <name>` not
      `start.sh stop`+`all` (which killed healthy bots). +3 regression tests. VERIFIED: bots survive past
      old 60s churn point.
- [x] 2.10 NEW: `_record_domain_success/_record_domain_failure` NameError FIXED (13 bare call sites → self.).
      Was firing every PDCA cycle (cost_gate_check_failed). +0 tests (covered by pdca suite). commit 9da748f6e.
- [x] 2.11 NEW: rate-limit `set lockMap` + authority-hunt `move` re-emission. Re-sending the SAME lockMap/
      hunt-move every autonomy cycle resets the client route task → re-triggers C A* → main-loop starvation
      (keepalive/snapshot starved → keep-alive churn → Perl heap grows to tens of GB). Gate both with per-bot
      rate limits (lockMap 30s, hunt-move 15s). commit e28a43453. RAM bloat root-caused to this churn.
- [x] 2.12 NEW: diagnosis — bots at spawn CAN route (BFS verified 7 cells izlude (127,253)→(125,257), all
      walkable) and position decodes CORRECT (map_changed 0091 = izlude 127 253). Snapshot wiring confirmed
      working via manual POST (state → izlude). Robots weren't sending snapshots because their main loop was
      saturated re-routing (churn), NOT an unwiring.
- [x] 2.13 NEW: MAP-AGNOSTIC academy-door resolution. Wired the dormant `_cold_start_academy_door`
      (data-driven: reads tables/portals.txt for the warp from the CURRENT map → iz_ac01) into the
      step-1 academy block, replacing the hardcoded `if _cs_map=="izlude"` + `move 125 257` literal.
      Fixes: (a) the helper referenced UNDEFINED symbols `_tables_root_cache()`/`AI_sidecar_base_dir`
      → NameError → always-None (silently broken); (b) hardcoded izlude. Verified: 'izlude'→'125 257',
      'prontera'→None. Also clamp stuck-detector random target to map bounds (was 'move -15 3'),
      gate portal-return + stuck-detector during cold-start steps<4, disable randomWalk during academy
      walk. Commits 47b2deab0/98ad436ac/0e66c3546. Root cause of the C A* spin + 74GB Perl heap:
      FOUR conflicting move emitters (academy vs portal-return vs stuck-random vs goal-decomposer
      'move prt_fild08') fighting every cycle.
- [x] 2.14 RESOLVED architecturally: single-destination authority (7857ee86d). The
      ~150 move emitters across ~25 modules can no longer stack conflicting moves:
      action_queue auto-assigns conflict_key='move' to every non-reflex
      move/navigate/move_random and the reserved key is LAST-WRITE-WINS (newest
      destination supersedes). Emitters/agents only PROPOSE; the queue is the
      single authority. Band-aid gates reverted (288354884). The bot reached the
      academy (iz_ac01) + completed the intro quest with this fix.
- [x] 2.15 CRITICAL memory leak fixed (d5e635cc6): PathFinding_DESTROY zeroed the
      session pointer before CalcPath_destroy -> every route calc leaked ~5MB
      (74GB observed). Verified: 300 create/destroy cycles flat at 52MB.
- [x] 2.16 COLD_START state-exit keying fixed (833c42f9a): _get_state read step with
      full bot_id vs stable-key writes -> never exited COLD_START.
- [x] 2.17 LLM NPC-dialog responder (355f24eee + cf3455250): conscious LLM reads the
      ACTUAL menu options (bridge pre/npc_talk_responses hook -> snapshot raw) and
      picks the response agnostically (option or free-text chat for AI NPCs);
      self-learns via server_solutions_store. Hardcoded academy 'r0' sequence
      REMOVED (founder: no hardcoded dialog solutions).
- [x] 2.18 LLM cold-start advisory (81c850951): conscious LLM decides the agnostic
      cold-start plan from FACTS (level/inventory/map/academy-warp/server solutions).
- [x] 2.19 Hardcoded 'move 22 203' (Prontera portal, non-walkable on izlude → A* spin)
      REMOVED from all fallbacks (f1f292368 + ec530e075) — the return-to-hunt block
      now skips during cold-start steps<4 and emits only when the knowledge DB has a
      REAL portal fact for the CURRENT map. Stuck-detector random move clamped to map
      bounds (3e28803f7 — was 'move 0 -10'). NPC/portal coords (weapon_shop,
      academy_receptionist, portal_to_town/hunt) seeded as knowledge-DB FACTS with
      x/y columns (+idempotent ALTER) — data-driven, never decision literals
      (f1f292368).
- [x] 2.20 Full suite 405+ passed (only the academy test was stale, now 8/8 green).
- [x] 2.21 REACHABLE-FARM resolution (4e1a78133 + c0e32f1c1): a level-1 bot locks a
      farm OpenKore can actually ROUTE to. Root cause: get_hunting_maps(1) returned
      [] (no level-1 maps in MAP_KNOWLEDGE) -> hardcoded fallback that may be
      unroutable from the current town. Added prt_fild08c/prt_fild05 level-1 entries +
      reachable_hunting_maps() (BFS over the bidirect portal graph, depth-2 cap —
      OpenKore chains only ~1-2 hops). izlude lvl1 -> prt_fild08 (1-hop); once in
      prt_fild08 -> re-lock prt_fild08c. prontera lvl1 -> prt_fild05. 9 transit tests.
- [x] 2.22 Portal DATA completeness (799945f36): prt_fild07 -> prt_fild08{a,b,c,d}
      reverse portals (OpenKore routes by edges; missing reverse = 'Cannot calculate
      a route'). Server-agnostic table data.
- [x] 2.23 PathFinding reset-path leak (8b443fada): CalcPath_init now frees the prior
      map buffer before re-calloc. VERIFIED in isolation (500 resets +4.9MB). NOTE:
      PathFinding.xs ALREADY freed on reset (lines 48-52) — the .cpp fix was
      redundant for the reset path; the DESTROY fix (d5e635cc6) is the real
      create/destroy-path fix (verified 300 objects flat at 52MB).
- [ ] 2.24 OPEN: live bot STILL leaks ~47MB/s steady (1.6GB -> 13.7GB while idle,
      even with no route calcs — only 0360 syncs). NOT pathfinding churn (reset path
      was already freed; idle = no CalcPath). Suspect: OpenKore CORE Perl main loop
      (getPacket/process) or bridge queue accumulation. NEXT: instrument
      PathFinding.xs with a live-object counter to PROVE it's not pathfinding, then
      audit OpenKore core loop buffers + bridge queues with gdb x/growth.
- [x] 2.25 Reachable-farm + portal data fixes verified live: bot locks prt_fild08
      (reachable 1-hop) from izlude, no 'Cannot calculate a route to prt_fild08c'
      decision failure. Bot warps into iz_ac01 + talks to Academy Receptionist
      (menu captured by bridge). Full progression to farming still blocked by the
      snapshot-completeness gap (sidecar sees base_level=0/hp=0/1 -> cold-start
      fires character-creation on the in-game bot).

## PHASE 3 — 0x0501 / PACKET COMPLETENESS
- [ ] 3.1 Implement the pending 0x0501 var-length recvpackets.txt patch (defensive; RAW already registers server-side).
- [ ] 3.2 Verify OpenKore tolerates unknown server packets (no warn/misparse) for all RAW-sent packet IDs.

## PHASE 4 — FOUNDER MANDATES (D1-D4, RULE.md, big pictures)
### 4a. P2P integration (D1/D2, RAW_P2P_INTEGRATION_PLAN)
- [ ] 4a.1 In-game P2P mesh (WebRTC data channel, 0x035F/0x0361) — bot joins mesh like RAW client.
- [ ] 4a.2 P2P relay registration + honest capacity.
- [ ] 4a.3 Peer-host map-server (bot hosts maps) — capacity node.
- [ ] 4a.4 IPv6 + UDP transport paths.

### 4b. Telemetry / ML crowdsource (I1-I4, P1-P2)
- [ ] 4b.1 Structured anonymous uploadable log stream (actions/decisions/outcomes/rewards/state snapshots).
- [ ] 4b.2 P2P crowdsource self-learning across bot peers (K1-K4, weighted-trust champion-gate).
- [ ] 4b.3 Prompt + agent-decision telemetry (redacted/bounded).

### 4c. Windows launcher integration (E1-E3, R1-R2, M1-M3, O1-O2)
- [ ] 4c.1 Build openkore-ai-v3 as single-file Windows .exe (bundles interpreter+deps).
- [ ] 4c.2 Launcher option to configure + run openkore-ai-v3 (account/char/server/ports/credentials auto from login).
- [ ] 4c.3 dist/ + manifest entries; OS-agnostic paths/processes/signals (J1-J3).
- [ ] 4c.4 Multi-char: launcher runs N instances, one per selected char.

### 4d. LLM key model (F1-F3, Q1-Q2)
- [ ] 4d.1 bot uses user's own LLM key / shared pool combos; 3 new combos.
- [ ] 4d.2 NO rewards for openkore-ai-v3 LLM usage.
- [ ] 4d.3 Openkore-AI User role vs LLM Token Supporter role logic.

### 4e. 4th-job end-game (L1-L3)
- [ ] 4e.1 Job-agnostic AI; long-term goal = 4th-job build; progression planner routes via real quests.

### 4f. ML / resource optimization (N1-N3)
- [ ] 4f.1 ML optional-but-enabled; GPU-first/CPU-second/disable; lowest resource usage.

## PHASE 5 — FULL-CLASS SWEEPS / VERIFICATION
- [ ] 5.1 Full Python suite green (335+ tests, from AI_sidecar cwd).
- [ ] 5.2 Perl suite green (1168, `make test`).
- [ ] 5.3 Whole-repo TODO/FIXME/stub/pass/dormant sweep (S19-S24 pattern), resolve each.
- [ ] 5.4 Reconcile sibling WIP (rathena char_clif.cpp 0-byte; procedural NPC files) — no conflict.
- [ ] 5.5 Commit after each batch; push at reasonable stage; update all tracker docs in each commit.

## VERIFIED LIVE NOW (baseline, 2026-08-25 08:10)
- Bot PID 1900355 in-game 10+ min, drop count frozen at 39, sidecar up (bot_count:2).
- RAW login/char/map all active (6900/6121/5121). Field data fld2 == GAT (correct).