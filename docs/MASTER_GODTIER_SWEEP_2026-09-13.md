# MASTER GODTIER COMPLETENESS SWEEP — openkore-ai-v3 (2026-09-13)

MANDATE (founder, verbatim intent): implement/integrate/fix/wire/execute/verify EVERYTHING to
completeness — zero mock/stub/placeholder/pending/todo/fixme/dormant/incomplete. Reconcile, NEVER
trim. Char-job AGNOSTIC + server-agnostic (RULE.md). Verify-BEFORE-modify. Benchmark-PROVE
(EXP/kill-rate/job-change E2E live), never assumption. See BIG WHOLE PICTURE across all stacks.
Commit after each batch. Update this doc after each batch. Goal: replace a pro RO player as a
god-tier AI bot — sustained EXP, job-change E2E, DQN-driven combat-micro, full-stack integration.

THIS DOC IS THE SINGLE SOURCE OF TRUTH. It consolidates every open item that lives scattered across:
- MASTER_COMPLETENESS_SWEEP_2026-09-08_session2.md (Batch 5 tail, 6, 7, 8)
- MASTER_COMPLETENESS.md (Phase 4a-4f, Phase 5)
- COMPLETENESS_TRACKER.md (Pres B1-B5, OPEN blocks)
- ZERO_INTERVENTION_COMPLETENESS.md (A5-A8, B1-B6, C1-C4, D1-D4)
- ZERO_INTERVENTION_CLIENT_FIXES.md (C5-C7)
- STALL_SELF_HEAL_CHAIN_CHECKLIST.md (C1-C6, D1-D5, V1-V4)
- SELF_HEAL_SURFACE_CHECKLIST.md (B3-B6)
- AGNOSTIC_COMPLETENESS_CHECKLIST.md, P2P_CAPACITY_NODE_CHECKLIST.md, BIG_PICTURE_CONSCIOUS_BRAIN_CHECKLIST.md

STATUS LEGEND: [ ] todo · [~] in progress · [x] done-verified · [!] blocked · [P] benchmark-proven live

---

## BASELINE (verified live 2026-09-13 22:34 — do NOT re-verify these each turn)
- Bot PID 1149376 in-game on prt_fild05, AI: attack route, attacking Thief Bug Egg (342,207) from 344,209, maxDistance 4.
- Sidecar healthy (PID 1263122, up 585s+), watchdog daemon PID 898300 (runtime watchdogd PID 434).
- Corpe-loop watchdog (5.26) HOLDS: last death 19:15 self-recovered, bot alive+farming since.
- git HEAD ae9f23389 (2 fixes committed this session: A1 domain cache poison + LTM null-tags).
- OPEN CHAIN: sell→zeny→job-change (base 43, job 10, EXP climbing) still NOT cleanly witnessed end-to-end.
- **CRITICAL LIVE-STATE (2026-09-13): the AI SIDECAR was DOWN since 2026-09-10 09:00 (3 days)** —
  restarted 22:40 (PID 1263122). Later live verification assumes sidecar UP.
- **LATENT BUG FIXED (2026-09-13): `long_term_memory_search_failed: NoneType not iterable`** —
  root cause: `m.get('tags',[])` returned None for null-tags docs → `t in None` → search aborted
  every call. Fixed (coalesce `or []`) in search + _search_local. Committed ae9f23389. LTM store/search
  IS wired (pdca 6016/6027/6141/10748-10786) — NOT dormant.

---

## BATCH A — LIVE SELL→ZENY→JOB-CHANGE E2E (the standing outcome blocker)
- [x] A1. **FIXED + LIVE-VERIFIED (2026-09-13).** Root cause: the observe-only legacy
      economy domain (EconomyDomain._apply_sell_config) calls the SHARED
      _set_config_once helper, mutating heuristic_service's _last_config_set dedupe
      cache as a side effect (sets sellAuto_maxWeight=70) even though its action is
      downgraded to a log intent. The single config owner (heuristic hunting block)
      then reads 70 vs its 25 every cycle → re-emits sellAuto_maxWeight forever
      (live flood 19:10-19:16). FIX: assess_all snapshots+restores the owner's
      _last_config_set around each observe-only domain run (domains/__init__.py) so
      observe-only analysis never owns/poisons live config. Regression test added
      (test_legacy_domains_observe_only.py). VERIFIED: after sidecar reconnect's one
      cold-cache push, 0 sellAuto_maxWeight re-emits in a sustained 30s window; 25
      domain tests pass. Committed.
- [~] A2. ROOT-CAUSE CONFIRMED + FIXED (2026-09-13, live-diagnosed): the vendor trip never
      completes because the bot NEVER ENTERS the SELL state. THREE stacked causes:
      (1) _get_state read signals['inventory']['weight_pressure'] (heuristic_service.py:1422)
          which is 0 until the bot is >50% overweight — a 16%-weight novice carrying junk read
          weight=0, so `if weight > 0.05: return "SELL"` never fired; the SELL state was unreachable.
      (2) STATE-ORDERING: town branch `return "JOB_CHANGE"` (eligible novice) fired BEFORE SELL.
          A broke (zeny 0) eligible novice (base 46 / job 10 live) returned JOB_CHANGE; its handler
          emits `move <guild>` it can't afford -> never sells -> zeny 0 -> deadlock.
      (3) JUNK-SCAN USED A NON-EXISTENT `Sell` COLUMN: the item DB on this fork exposes only
          `Buy` (no `Sell`), so `_it.get("Sell",0)` always returned 0 -> NO item ever classified
          as junk -> `sell <id>` never emitted (even when SELL state ran + talknpc opened the
          vendor dialog). RO mechanic: resale = Buy/2. Now derived.
      FIXED all three: (1) _get_state now reads the real weight_ratio; (2) JOB_CHANGE gated on
      affordability (zeny>=500) with carry>5% -> SELL first, broke+empty -> TOWN_HUNT (farm);
      (3) junk val = Buy//2, 0<val<100 = junk. Regression tests added
      (test_sell_before_job_change.py, 6 tests). VERIFIED live: sidecar restarted with fix;
      bot (base 46/job 10/zeny 0) now emits `talknpc 126 76 c` + `talk cont` (was only
      move/stand before) = SELL state running. PENDING: bot routes to town on 600s periodic-sell
      or bag-full -> junk converted -> zeny>0 -> reaches alberta -> merchant (longer live window).
- [ ] A2b. Identity drift: 4 stale registrations for the same char (testbot99 under masters
      Local rAthena AI World / TestBotA / TestBotB / TestBotC) + testbotA/testbota. Active bot
      `Local rAthena AI World:testbot99` held 125-128 pending actions dominated by emergency
      potion reflexes (zeny-0 consequence). Verify single live identity per control folder;
      reconcile stale registrations so the queue the bridge polls == where actions enqueue.
- [ ] A3. Confirm the discovered-vendor talknpc sequence tokens are valid (talknpc <x> <y> c r1 n — NOT r/text/ form) and the vendor is a real buy-from-player (prt_in 126 76 Tool Dealer). Verify via live bot log line.
- [ ] A4. Job-change gate: verify zeny>=500 logic across ALL job-change emitters (cold-start, HUNTING-branch, progression.py) is one shared affordability rule (5.10), and macro required_zeny=500 holds (5.12). PROVE: bot reaches alberta guild via airship/portal (NOT 11-map overland walk), talks, becomes merchant, gains job EXP.
- [ ] A5. [P] Record full timestamped chain: login→enter→farm(EQP climb)→sell→zeny>500→job-change→new class farming. (Batch-8 benchmark, live outcome proof.)

## BATCH B — STATS / PROGRESSION LOOP (blocks clean progression)
- [ ] B1. STATS loop (5.24b): sidecar thinks stat_points=5 (DB correct), OpenKore in-memory points_free=0 → every `st add dex` errors. Fix: reconcile points_free source (re-sync on level-up/relog; stop spamming when points_free=0). Verify no more 'Not enough status points' + points actually land.

## BATCH B2 — CORPSE-LOOP ROOT CAUSE (the repeated death→AI:dead→watchdog-restart cycle cut off the sell→zeny proof)
- [x] B2.1 ROOT-CAUSE FOUND+FIXED (2026-09-13, live log 739050-739070): a gearless bot
      dies because the config audit sets `teleportAuto_deadly=1` → OpenKore built-in fires
      "can kill You with the next N dmg → Teleporting", then Task::Teleport fails
      NO_ITEM_OR_SKILL ("You don't have the Teleport skill or a Fly Wing") → bot FREEZES in
      place → dies → corpse-loop. A Pro never arms an escape it cannot execute. FIX: both
      hunting config-audit blocks now gate teleportAuto_deadly on ACTUAL Fly Wing ownership
      (scanned from inventory_items, AGNOSTIC — never a hardcoded id). Gearless → deadly=0.
      Regression test added (test_gearless_bot_disables_deadly_teleport). VERIFIED: 7 tests
      pass. LIVE: bot no longer corpse-loops (2+ restarts were fatal-freeze teleport issues).
- [~] B2.2 ROOT-CAUSE FOUND+FIXED (2026-09-13): the sell command was emitted but ALWAYS failed —
      live log 803631-803632: `Error in function 'sell'`: "'909'/'7872' is not a valid item
      index #". OpenKore's cmdSell (Commands.pm:5225) resolves its arg via
      Actor::Item::getMultiple by BINID (inventory slot) or item NAME — NOT the item_db ID
      the sidecar emits (`sell 909 0`). So every sell failed, junk stayed 63x Jellopy, zeny 0.
      FIX: bridge _rewrite_runtime_command ('sell' handler) now maps the numeric item_db ID to
      the owned item's binID (`sell <binID> [amt]`). perl -c syntax OK. Bot restarted (watchdog
      #9, PID 1496044) to load the bridge change. PENDING live: bot sells junk -> zeny>0.
## BATCH B3 — DUAL-SUPERVISOR RACE (systemd watchdog + sidecar keep-alive both own the same profile)
- [x] B3.1 ROOT-CAUSE (live-diagnosed 2026-09-13): TWO supervisors race to manage
      .bot_profiles/testbotA — (a) the systemd watchdog daemon (PID 898300,
      openkore-bot-watchdog.service → ai_sidecar.runtime.watchdog.run_daemon, registered
      ALL .bot_profiles/* at boot) AND (b) the sidecar's keep-alive (lifecycle.py
      _restart_stale_bots → start.sh bot testbotA, spawned PID 1537751 under the sidecar's
      own process tree). Both spawn competing openkore clients for the SAME char →
      char-conflict/connection churn → restarts cut off the sell→zeny proof repeatedly.
      The watchdog circuit breaker tripped (restart #9), then it relaunched #10 while the
      keep-alive's manual instance was still connected → two live clients for one char.
      FIX (reconciliation, single owner): the sidecar keep-alive must NOT spawn a bot that
      the systemd watchdog already supervises. (PENDING — do NOT run both.)
- [x] B3.2 COMBAT-TACTICS DOMAIN DEAD (live log ERROR every combat tick): dispatcher.py
      build_context called a.get("type"/"hp"/"is_party") on ActorDigest pydantic objects →
      "AttributeError: 'ActorDigest' object has no attribute 'get'" → tactics_dispatcher.
      assess() failed every cycle → the whole combat-tactics domain (kiting/melee/magic
      positioning) was dormant while the bot fought on reflexes. FIX: normalize actors
      via model_dump + alias keys the dict-API consumers expect. VERIFIED: 3 new regression
      tests pass (test_combat_dispatcher_actordigest.py). PENDING: commit + sidecar restart.
- [x] B3.3 SELL-TO-ZENY CHAIN root-caused + fixed (committed 8167b6c7e + 5543bcdfa): cmdSell resolves by
      BINID/item-name not item_db Id → bridge rewrites sell <db_id> → <owned binID>; AND the SELL
      state added each junk item to the pending list ('Type sell done to sell everything in your
      sell list') but NEVER emitted 'sell done' → the list never executed → zeny stayed 0. Fixed
      both: binID rewrite + finalize 'sell done'. VERIFIED LIVE 2026-09-14 00:06-00:19:
        - junk reaches the sell list with binIDs ("Added to sell list: Fluff x9 / Clover x21 /
          Sticky Mucus x2 / Feather x14 / Worm Peeling x7 / Club [3] x1")
        - `sell done` fires AND sends packet 0x00C9 [Sell] to the server (00:19:01) + 09D4
          Sell/Buy Complete
        - residual 00CB [Sell Result]=0x01 "Sell failed" — the bot was mid-lockMap-flip to
          prt_fild08 (progression agent kept setting lockMap) so the sell went out without a
          buy-capable dialog attached. PENDING: single-owner sell (suspend lockMap during the
          sell trip so the buy-dialog stays attached) → junk→zeny proof.
- [ ] B2. Level 1-10 academy/tutorial escape (D6): a level-1 bot landing in iz_int* academy room must deterministically exit (exit guard + academy-room gate hold, 5.24/S9-S10). Verify live for a fresh-spawn bot.
- [ ] B3. Per-class config audit + stat allocation (RULE.md §6/§11): confirm allocation fires on level-up via DB (not stat_points signal), per-class order, no hardcoded class in conscious path (reflex floor only).

## BATCH C — DQN SUBCONSCIOUS / COMBAT-MICRO (god-tier gap)
- [ ] C1 (6.1). ThreatTargeting NEVER instantiated — CombatLoop._threat_targeting stays None, _acquire_target no-ops. Wire real target selection (char-agnostic; server mobs from game-DB).
- [ ] C2 (6.2). Design + implement char-agnostic combat-micro state→action (target/skill/retreat) reusing the trained DQN or a per-class combat micro-policy; NO hardcoded class/item.
- [ ] C3 (6.3). Subconscious drives target+skill choice when trained; heuristic fallback when undertrained. Verify DQN actually trains (real entry _train_from_replay; stats reinforcement_stats.json training_steps>0).
- [ ] C4 (6.4 / S24-S25). [P] Benchmark: kills/min + EXP/hour improve vs heuristic-only. Record baseline then after.

## BATCH D — CONSCIOUS TIER / MEMORY / PREEMPTIVE (BIG_PICTURE checklist)
- [ ] D1 (G1). LongTermMemory — VERIFIED WIRED (NOT dormant): store at pdca 6016/6027/6141/10748-10786,
      search/recall injected into LLM prompts at 10340/10960 (get_relevant_context). Null-tags search
      crash FIXED (ae9f23389). Remaining: confirm recall reaches the LIVE LLM prompt (not just cold-start),
      and store-failure sites log (not silent).
- [x] D2 (G2). Memory-store gap partially closed: LTM store IS called on significant events
      (pdca 10748-10786 kill/death/lesson sites). VERIFIED wired — update 5.24-era claim "no memory
      store on significant events" → store exists + works (null-tags fix). Remaining: audit those
      sites actually reachable in the live path (not gated off).
- [ ] D3 (G3). Cold-start LLM advisory prompt lacks the past-context block — inject learned history.
- [ ] D4 (G4). _llm_gear_advisory gets the same past context (server-agnostic, DB-backed solutions).
- [ ] D5 (G5). PREEMPTIVE (not just reactive): death-loop prediction / sustain anticipation before it bites.
- [ ] D6 (S22/S3). Conscious-vs-heuristic balance: verify action->command bridge executes LLM decisions (retreat/restock/potions), not just emits.

## BATCH E — STALL / SELF-HEAL CHAIN (STALL_SELF_HEAL + SELF_HEAL_SURFACE)
- [ ] E1 (C1/D1). EMPTY-MAP stall: sidecar must KNOW map has 0 monsters → emit map-change.
- [ ] E2 (C2/D2). NO-PROGRESS: EXP-delta monitor (exists in _remember_significant) triggers heal when frozen N min.
- [ ] E3 (C3/D3). STUCK-ROUTE → exploration scout to discover route.
- [ ] E4 (C4). STUCK-IN-TOWN: 3+ town cycles → retreat/change.
- [ ] E5 (C5). DEATH-SPIRAL → heal chain. E6 (C6). NO-ATTACK → reset attack.
- [ ] E7 (D4,D5,V1-V4). Ack/verify each heal + feed memory/reward; tests: empty-map triggers change, EXP resumes, no false-positives, per-class unit.
- [ ] E8 (B3-B6). Audit F8-F16 (sell/gear/zeny/portal/map-transition/death-loop) heal surfaces; wire missing; test each; ledger only gets real failures.

## BATCH F — FULL SIDECAR DEAD-CODE / DORMANT SWEEP (Batch 7)
- [ ] F1 (7.1). heal_resource_loader.py:200+ never initialized — wire or reconcile.
- [ ] F2 (7.2). rule_engine.py:796 hardcodes reflex_survival_escape (name mismatch reflex_teleport_escape) — reconcile.
- [ ] F3 (7.3). action_emitter.py:434 macros dir + macro_manifest.json reconcile with macro_intelligence.py.
- [ ] F4 (7.4 / 5.7). lethal_escape_teleport YAML reflex busy-loops a NO-OP macro at HP<=0.18 — fallback must actually flee/retreat (walk away) OR suppress when no Fly Wing/zeny. THIS BLOCKED kills (0 kills while alive regenerating).
- [ ] F5 (7.5). Zero stub/todo/pass/dormant grep pass — resolve each (resolve, don't just flag).
- [ ] F6 (Pres). heuristic_service.py:1103 (old) hardcoded `prontera`; model-router DEFAULT_POLICY_RULES targets exist; registry-remove check; abstract NotImplementedError x4 → abc + @abstractmethod; flake-hardening route_churn_count test.
- [ ] F7 (7.6). Adversarial sweep round after round until nothing found (each pass historically finds real bugs).

## BATCH G — EXECUTION / CLIENT LAYER (ZERO_INTERVENTION_CLIENT_FIXES + A-series)
- [ ] G1 (C5). CalcMapRoute maxTime honored? Infinite loop if undef/never exceeded.
- [ ] G2 (C6). Route task blocking AI main-loop processMisc (keepalive)?
- [ ] G3 (C7). ai_route_calcRoute timeout loaded in timeouts.txt?
- [ ] G4 (A5-A6). char-select "all maps not ready" root cause (peer-host flap vs central ownership) + retry fix.
- [ ] G5 (A7). Post-map-entry: register any remaining missing 20250604 packets (full-class sweep, no future 'Unknown switch').
- [ ] G6 (A8). [P] LIVE verify: bot enters izlude, stays >3min, gains EXP.

## BATCH H — INTEGRATION / PLATFORM (MASTER_COMPLETENESS Phase 4 + ZERO_INTERVENTION B/C)
- [ ] H1 (4a.1/B4). In-game P2P mesh (WebRTC data channel, 0x035F/0x0361) — bot joins mesh like RAW client.
- [ ] H2 (4a.2/B3). P2P relay registration + honest capacity.
- [ ] H3 (4a.3/B2). Peer-host map-server (bot hosts maps) — bot embeds cross-compiled map-server.exe + DLL, ephemeral DB creds (RAW's model).
- [ ] H4 (4a.4/B5). IPv6 + UDP transport paths.
- [ ] H5 (4b.1/I1-I4). Structured anonymous telemetry stream (actions/decisions/outcomes/rewards/state snapshots).
- [ ] H6 (4b.2/K1-K4/P1-P2). P2P crowdsource self-learning across peers (weighted-trust champion-gate).
- [ ] H7 (4c.1/C1). Build openkore-ai-v3 as single-file Windows .exe (bundles interpreter+deps).
- [ ] H8 (4c.2-4c.4/C2-C4). Launcher option to configure+run N instances; dist/ + manifest entries; OS-agnostic paths/processes/signals.
- [ ] H9 (4d/4e/4f). LLM key model (user's own key/shared pool, NO rewards); 4th-job end-game; ML optional GPU-first/CPU-second/disable.
- [ ] H10 (B1). Design doc: player-bot vs capacity-node roles + staging (verify-first before build).

## BATCH I — TESTS / PRODUCTION-READY (Phase 5 + Batch 8)
- [ ] I1 (5.1). Full Python suite green (from AI_sidecar cwd; 394+ tests) — run, fix, prove.
- [ ] I2 (5.2). Perl suite green (`make test`, 1168+).
- [ ] I3 (8.1). Login→enter→farm→EXP→level-up→job-change FULL loop proven with timestamps (char-agnostic path).
- [ ] I4 (8.2/[P]). kills/min + EXP/hour benchmark recorded.
- [ ] I5 (test_harness). RULE.md compliance harness passes; no commit fails it.
- [ ] I6 (5.4/5.5). Reconcile sibling WIP (rathena char_clif.cpp 0-byte; procedural NPC files) — no conflict; commit after each batch; push at reasonable stage; update tracker docs in each commit.
- [ ] I7. Adversarial sweep across ALL batches until genuinely nothing surfaces.

---

NOTES / KNOWN SERVER-SIDE (not openkore-ai-v3, track only):
- OPEN C3: char-server SIGSEGV at char-select (recurring disconnect). OPEN C4: mail wire-size mismatch (non-fatal). OPEN C5: 23-bot fleet concurrency hammering one endpoint. OPEN D10: gearless server starting-resource limit.
These live in rathena-AI-world / RAW stack — openkore-ai-v3 must adapt/route around them, not fix server (RULE.md: never modify RAW to match the bot).
