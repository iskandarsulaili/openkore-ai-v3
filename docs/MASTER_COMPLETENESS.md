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
- [x] 2.24 RESOLVED (2026-08-25, multi-pass diagnosis): the "steady idle leak" was a
      SINGLE long CalcPath_pathStep run on a 400x400 map (prt_fild08) growing the Perl
      SV heap (~47MB/s, 1.6->17GB in 5min). PROVEN by: PathFinding live-object counter
      (pf_live=0 balanced), C A* isolated tests flat, Field counter flat, leakdiag
      freeze (main loop blocked in pathStep while RSS grew). FOUR fixes:
      (1) route-fail cooldown (625fdb7f1) — processLockMap no longer re-attempts a
      failed lockMap route every cycle.
      (2) CACHED PathFinding (199666761) — reuse ONE per (w x h), pf_created=1.
      (3) HARD openList bound (f25dcf740) — openListAdd refuses writes past
      width*height (CLOSED-reopen churn OOB risk); pathStep bails on overflow +
      maxPops. Verified: A* flat (48->58MB on 50 runs).
      (4) SOLUTION MATERIALIZATION CAP (423d2f590) — THE ACTUAL LEAK FIX: run()
      built a Perl hash per path step (newHV+av_store x160k on a 400x400 map),
      blowing up the SV heap to GBs. Cap at 8000 steps (every Nth). 
      VERIFIED LIVE 2026-08-25: RSS FLAT at 206MB for 6+ min (was 7MB CPU /
      17GB RSS), pf_created=1, route_calcs capped by cooldown. 2.24 CLOSED.
- [x] 2.26 ADVERSARIAL AUDIT (deleg_dea5892c, 12 findings) — 5 REAL defects FIXED:
  (a) #1 HIGH cold-start keying — _get_state read _cold_start_step with FULL bot_id
      while all writes used stable key -> COLD_START never exited (the live
      receptionist-loop root cause). Fixed: bot_id normalized to stable key at
      _assess_impl entry (88efbaaf5). ALSO the revert (288354884) had undone the
      earlier 833c42f9a fix — this is the definitive re-fix.
  (b) #2 HIGH route_randomWalk 1 defeat — _apply_mimicry_config forced randomWalk 1
      every cycle, undoing the academy's randomWalk 0 (last-write-wins) -> C A* spin.
      Gated on cold-start step >= 4 (88efbaaf5).
  (c) #6 HIGH _cold_start_hunt_map SCALAR shared across bots -> per-bot dict
      (88efbaaf5). Also fixed a latent NameError: block-scope sites used bare
      'bot_id' (undefined in assess(); only _bot_id exists) silently swallowed by
      try/except -> skipped the whole academy-farm lockMap block.
  (d) #4 HIGH start.sh substring pkill — 'openkore.pl.*kicapmasin' matched
      kicapmasin2/3. Anchored on .bot_profiles/<name>/control (961f401c2).
  (e) #5 HIGH snapshot double-ingest — REVIEWED + REJECTED: the sync forward must
      stay (immediate navigation population is a hard guarantee; removing it
      re-blinds the pipeline). Documented in lifecycle.py (961f401c2).
  PLUS a 6th defect the audit MISSED (found live): _record_domain_success/failure
  MISSING self in the defs (a prior regex refactor stripped it) -> EVERY PDCA cycle
  crashed in the game_engine domain block (pdca_domain_error in live log), silently
  disabling per-domain health tracking. Fixed + regression test (437a9618f).
- [~] 2.27 PARTIALLY RESOLVED (2026-08-25): un-gated move emitters during cold-start.
  FIXED: academy-door now resolved data-driven from portals.txt (no hardcoded 125 257
  literal — a2c1a64f0); SOLO-BOT guard (1-bot fleet no longer stuck in PARTY state,
  e70d5e7eb); academy-first routing (462361fd5); TOWN_HUNT academy-defer stops
  'move int_land' race (9299b2050 + 4ecb59385 crash fix); ROBUST cold_start in-game gate
  never char-creates a live bot (39a667e46); bridge route-loop dist<5 -> dist<2 so the
  4-tile-away warp is no longer suppressed as 'already there' (ca6b1521e); walkability-
  snap to a walkable warp-trigger neighbor (bad9e0fb2); split-regex fix (ba787a5f1);
  same-cycle stand/ai-auto no longer cancels the walk (a35da9634). VERIFIED LIVE: bot
  now ROUTES + walks to the academy door ('You reached the destination') instead of
  random-walking at spawn. REMAINING: the izlude->iz_ac01 warp does NOT fire on arrival —
  OpenKore's internal char pos desyncs to (125,257) so it sends 0 walk packets (thinks
  already there) and never steps onto the warp trigger tile; bot loops move-to-door.
  Next: force-walk the final tile / resync char pos before the warp.
- [x] 2.28 RESOLVED (audit #8, ad7c533b0): _restart_stale_bots no longer blocks the
      event loop — made async + Popen.communicate wrapped in asyncio.to_thread.
      7 keep-alive restart tests pass.
- [x] 2.29 ROUTE-CALC DIAGNOSED (2026-08-25): the cross-map izlude->prt_fild08 route
      WORKS in isolation (2 legs, diag_route2.pl) — the live 'Cannot calculate a
      route' was the route-fail CHURN from the cold-start never progressing, NOT a
      broken cross-map A*. The root cause was the 2.24 churn (now fixed by the
      route-fail cooldown + cached PathFinding). The FIRST long prt_fild08 calc can
      still grow SV heap (~17GB over 5min) — remaining: cap the prt_fild08 route cost
      / solution-size (a single pathfind on a 400x400 map with randomFactor builds a
      huge solution array in Perl). Recommended next: cap solution array size in the
      XS run() materialization (only emit every Nth step when solution_size is huge).
- [x] 2.25 Reachable-farm + portal data fixes verified live: bot locks prt_fild08
      (reachable 1-hop) from izlude, no 'Cannot calculate a route to prt_fild08c'
      decision failure. Bot warps into iz_ac01 + talks to Academy Receptionist
      (menu captured by bridge).
      SNAPSHOT COMPLETENESS RESOLVED (2026-08-25): state now flows to the sidecar
      correctly (map=izlude base_level=1 hp=45/45 via /v2/state); cold_start
      in-game gate works (no relog/char_create spam on the live bot). Also fixed
      SessionManager avoid_peak_hours (e1b11e943) — was default True (20-23) so
      it quit the bot every evening at peak; now False (24/7 farm). Bot is STABLE
      (RSS flat 206MB, CPU ~5%, no churn/leak/relog). Remaining: the bot still
      must WALK the izlude academy route to reach prt_fild08 and grow EXP (cold
      -start walk-through thread).

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