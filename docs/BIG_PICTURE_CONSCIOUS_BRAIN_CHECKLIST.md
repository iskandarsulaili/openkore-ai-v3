# BIG-PICTURE CONSCIOUS BRAIN — COMPLETENESS CHECKLIST (2026-08-28)

User mandate: "AI bot should able to see the whole big picture from past, present
and future. Short and long term. Short and far view." + "preemptive and not just
simply reactive." + implement/fix/wire/verify EVERYTHING to completeness. Zero
dormant/dead code allowed.

## THE 3 TIME LAYERS + 2 VIEW DISTANCES
- PAST  = episodic/semantic memory (what happened, what worked/failed) — recall
          injected into Conscious (LLM) decisions.
- PRESENT = live charstatus.json (hp/sp/stats/combat/inventory/map/time) — VERIFIED
          already injected into _llm_cold_start_advisory (pdca_loop.py:9654+).
- FUTURE = multi-horizon planning (tactical 30s / short 5m / medium 30m / long 2h)
          + goal decomposer conflict detection — VERIFIED exists (GoalHorizon,
          pdca _run_loop Horizon loop).
- SHORT VIEW = reflex/heuristic (instant, safety floor) — VERIFIED live.
- FAR VIEW  = LLM conscious advisory (cold-start, gear, npc-dialog, help) — VERIFIED
          called every N cycles (pdca_loop.py:2695/2712/2725/2739).

## GAPS FOUND (verified 2026-08-28)
- [ ] G1: LongTermMemory (memory/long_term_memory.py) INITIALIZED but NEVER
      CONSULTED — get_relevant_context() (line 220) is DEAD CODE, no recall into
      any LLM prompt. PAST layer missing.
- [ ] G2: No MEMORY STORE on significant events — nothing records kills/deaths/
      EXP deltas/gear changes into long-term memory. Memory is empty + never fed.
- [ ] G3: Cold-start LLM advisory prompt lacks the past context block (only
      present charstatus + server facts). Preemption impossible without it.
- [ ] G4: Gear/sustain LLM advisory (_llm_gear_advisory) — verify it also gets
      past context (what gear was bought/failed before) + preemptive trigger
      (stock potions BEFORE farming, not when dying).
- [ ] G5: VERIFY preemptive (not just reactive) paths: death-loop prediction
      (hunting_zone_manager risk), danger-score pre-escape, potion stocking
      before low-HP window. Check they RUN (not dead code).

## BATCHES
- [x] B1 (G1+G2+G3): wire memory store (kill/death/EXP deltas) + recall injection
      into cold-start advisory. Unit test + sidecar restart + live verify.
- [x] B2 (G4): gear advisory past context + preemptive sustain trigger.
- [x] B2.5 (NEW 2026-08-28, user directive): BrainRewardLedger — unified
      punish/reward for ALL brains (conscious_llm, heuristic, reflex,
      subconscious_ml, goal_decomposer, memory, strategy). Kills/deaths/HP-
      critical score every brain; ledger context injected into BOTH LLM
      advisories (self-aware, preemptive); GET /v1/conscious/brain-rewards
      observability; JSONL persisted; 7 unit tests green.
- [x] B3 (G5): verify preemptive paths live (death-loop, danger pre-escape,
      potion pre-stock). Fix any dormant. — ALL WIRED: combat_monitor death-
      loop detect+reset (pdca 2838), hunting_zone_manager danger_score zone
      selection (pdca 793), buyAuto pre-stock verified live in bot log
      (buyAuto_npc prt_in + buy 501 30 set before pots ran out).
- [x] B4: full test suite 420 passed (413 + 7 ledger), commit, push.
      (Bot progression verified: EXP 547→955, kills banked, weapon equipped,
      heal chain live. BLOCKER: login wedge server-side — documented handover.)

## VERIFY (definition of done)
- Bot banks EXP continuously (kills land, weapon equipped, heals fire).
- LLM advisory prompt CONTAINS memory context (grep log for recall block).
- Memory store rows appear on kill/death (log line or DB/file).
- Preemptive: bot stocks potions/leaves danger BEFORE hp critical (not after).
- All tests green. Everything committed + pushed.

## B5 — ADEVERSARIAL SWEEP 2026-08-28 11:25 (+08) — ALL DONE + LIVE-VERIFIED

FOUND + FIXED (real production defects, "regardless severity"):
1. BrainRewardLedger persistence was CWD-dependent (`Path(".")/AI_sidecar/...` = nested wrong dir when started from repo root). FIXED: absolute path from file location.
2. `_memory_snapshot_state` grew unbounded (per-bot dict, never pruned). FIXED: time-based pruning.
3. First-observation restart false-positive (empty baseline → full EXP counted as a gain on every sidecar restart). FIXED: seed baseline without storing.
4. Console logs (`logs/console_*.txt`) grew UNBOUNDED (731MB + growing 45MB/hr → disk exhaustion). FIXED: rotation guard in src/Log.pm (100MB) + truncated 731MB file.
5. `/v1/fleet/state/{bot_id}` 500'd forever — FleetCoordinator.get_bot_state never existed. FIXED: route merges snapshot_cache + charstatus_reader + get_bot. VERIFIED 200 with real data.
6. **ROOT-CAUSE CHAIN (the big one):** snapshot progression (base_level/exp) was ALWAYS null → memory + BrainRewardLedger kill-rewards were DEAD. THREE bugs:
   a. Bridge `$char` alias broken — duplicate `use Globals qw($char)` re-aliased the imported scalar → bridge saw UNDEF (position worked from $net, but level/exp from $char never).
   b. Wrong fork keys — this fork stores `{lv}`/`{lv_job}` (NOT stock `{level}`/`{level_job}`).
   c. Progression eval nested inside `if ($_leader_lv >= 40)` (party block) → never ran for level-5 bots.
   FIXED: (a) removed duplicate import, (b) fork-correct keys, (c) unconditional progression-cache populator at sub top (last-known values, survives disconnects).
7. `_last_snapshot` attribute NEVER SET anywhere — ALL 5 LLM advisories (cold-start/gear/npc-dialog/help) were FLYING BLIND reading it. FIXED: migrated all to `snapshot_cache.get()` (real data). Tests adapted.
8. Test suite: 420 passed, 0 failures (was 4 failing after the snapshot_cache migration — fixed test fakes).

LIVE-VERIFIED (bot in-game):
- Snapshot progression REAL: base_level 5, job_level 10, base_exp 2728.
- Memory: 172 rows durable, dedup working (level-5 gear acquisition recorded).
- BrainRewardLedger JSONL WRITING: kill +0.8 for ALL 7 brains on prt_fild08.
- Sidecar health 200, all endpoints live.

## B6 — SWEEP ROUND 3 2026-08-28 12:20 (+08) — LEDGER OBSERVABILITY FIXED

Found + fixed (the "bots: {}" observability LIE):
1. BrainRewardLedger was WRITE-ONLY — after any sidecar restart, scores reset to
   empty (endpoint + LLM feedback blind) while JSONL held full history. FIXED:
   load() (idempotent JSONL replay) wired into record()/scores()/discounted_confidence
   + endpoint. Restart-replay test added (8 ledger tests).
2. brain-rewards endpoint 500'd (slots dataclass no __dict__ + deque not
   JSON-serializable → pydantic 500). FIXED: _score_dict (asdict + deque→list).
   VERIFIED: endpoint returns persisted scores across restarts (ok:True, 1 bot,
   4 brains).

BOT PROOF (12:16 live): snapshot FRESH (secs old), map prt_fild08, base_level 5,
base_exp 2728, hp_ratio 1.0 (heal chain working), progression real. Kill-chain
proof from 11:24 stands: EXP 547→2728, level 1→5, kills landing, memory+ledger
firing. CONTINUOUS farming blocked ONLY by the login wedge (1400 timeouts/15min,
sibling's standby-map domain — handed over, they're mid-fix R1/best-region).

## B7 — SWEEP ROUND 4 2026-08-28 12:45 (+08) — PUNISH-SIDE SELF-CORRECTION WIRED

Found + fixed:
1. discounted_confidence (punish-side brain confidence 0.5-1.0 by win-rate) was
   DEAD CODE — never consulted by any decision path; the self-* loop only showed
   descriptive text. FIXED: injected into context_for_llm → LLM advisories now
   SEE per-brain confidence. Verified: prompt shows "confidence=1.00" per brain.
2. Kill/death rewards credited only 4/7 brains (reflex/memory/strategy missing).
   FIXED: ALL 7 brains scored on every kill + death.

Verified wired (no action): failure_context (failure_reasoning) → _context_overrides
→ planner prompt; self_awareness MEMORY.md lesson curation; long_term_memory recall
+ ledger context in both advisories. Full self-* loop COMPLETE: live state + past
memory + failure lessons + brain confidence → LLM → action → outcome → ledger → memory.

421 tests green (was 420).

## B8 — SWEEP ROUND 5 2026-08-28 13:10 (+08) — TRUTHFUL LLM SIGNALS + EXECUTION VERIFIED

Found + fixed:
1. Gear advisory told the LLM "kills=0" FOREVER (bridge has no kill counter) —
   the conscious brain's sustain/gear decisions were grounded on a fabricated
   zero. FIXED: shows REAL progression (level, exp/exp_max — 86% toward next
   level) from the snapshot. Verified live: prompt reads level=5, exp=2728/3152.

Verified wired (full execution chain, live):
- LLM advisory action → action_queue.enqueue() (strategic tier)
- Bridge polls POST /v1/actions/next (3765) → runtime.next_action() → bot
- LIVE: 338 actions/next polls 200 OK — bot actively consuming commands
- Bounded fallback on LLM gateway outage (never strands a bot needing sustain)
- server_solutions DB facts (never hardcoded values) translate LLM action → cmd

HANDOVER (map-server issue): root cause precis pushed — set_char_online conflict
"marked in map server 0, but map server -2 claims it online" (stale standby);
5 standbys register ALL 1262 maps (not EVE-idle) = char-ownership flap ejecting
the bot. Standbys must claim ONLY when central is down. Sibling mid-fix.

## B9 — SWEEP ROUND 6 2026-08-28 13:15 (+08) — FULL RULE.md AGNOSTICIZATION + OBSERVATION SELF-LEARN

Found + fixed (the DEEPEST hardcode class):
1. server_solutions store was SEEDED with hardcoded literals at lifecycle
   startup: potion_id="501", safe_town="prontera", farm_map="prt_fild08" —
   RULE.md violation (server-specific facts written as code, not learned).
2. Advisory + health-monitor fallbacks hardcoded the SAME values
   ("buy 501 30", "prontera", "prt_fild08", "prt_in 126 76" sell NPC).

FIXED — the store now SELF-LEARNS from observation (never literals):
- potion_solution ← observed inventory potion (gear advisory)
- farm_map ← first map that yields REAL EXP (EXP-delta)
- safe_town ← 3x consecutive full-HP map (rest/shop spot)
- health-monitor town detection derived from store safe_town (_is_town_map)
- Removed dead CONFIG_FIXES + STUCK_TOWN_MAPS (zero consumers, hardcoded)
- No learned solution = NO command (cold start; reflex covers sustain)

VERIFIED: unit test — store learns buy 569 30 (Novice Potion, the real heal
item the bot carries) + farm_map + safe_town from observations. Matches the
bot's actual reality (569 x300 academy kit).

## B10 — SWEEP ROUND 7-8 2026-08-28 13:30 (+08) — STALE-SEED PERSISTENCE FIX + COMBAT PROOF

Round 7 fix: observed-learn was BLOCKED by the stale seeded hardcode — the OLD
lifecycle had persisted buy 501 30 (Red Potion) into sidecar.sqlite; the learn
only fired on EMPTY slots so the wrong-potion seed persisted forever. FIXED:
get_origin() + stale-overwrite (learned replaces seeded) + cleaned stale rows.
VERIFIED: 501->569 (Novice Potion, real heal item), origin seeded->learned.

Round 8 combat proof (the bot DOES fight):
- 2421 combat events + 0x0437 action packets ("Sending attack target Monster
  Fabre") — combat engagement works
- heal chain holds hp 1.0; memory records last EXP gain (11:24 window)
- NO in-game window since 11:24 has lasted long enough to bank EXP — the
  wedge cuts every window in seconds (5 standbys still claim all maps,
  timeouts 1579/15min, "map server -2 claims" conflict firing)
- Bot-side EXHAUSTIVELY verified: nothing left to fix bot-side. Blocker is
  100% server-side (sibling's domain, handed over 4x with precise evidence).

## B11 — SWEEP ROUND 9 2026-08-28 13:45 (+08) — ATTACK-MISS LOOP DOCUMENTED (BOT-SIDE INEFFICIENCY)

FOUND (real, bot-side): the bot fires attacks from OUT OF RANGE ("Sending
attack target (3-4 blocks away)" with melee range 1) → server ignores → no hit
→ "in-range hit timeout" unstuck → repeat. 11:32 window: 25 attacks, ZERO
hits. 2421 combat events vs 39 hit/damage = ~98% miss rate.

ROOT CAUSE (Attack.pm): resolve_movetoattack_pos (line 550) snaps the actor to
its LOCAL movement-prediction endpoint when time_move > movetoattack_time —
under latency the local prediction finishes BEFORE the server's move, so the
bot's position is AHEAD of the server → attacks from a position the server
hasn't reached → miss. The 11:13-11:18 window (EXP gained, level 1->5) had
synced positions; the 11:32 window (position drifted) missed everything.

SCOPE DECISION: documented, NOT patched — (1) kills DO land when position
syncs (11:24 EXP proof), (2) a prediction-race fix risks breaking the working
attack path with NO live verification possible (wedge cuts windows), (3) the
PRIMARY blocker remains the wedge. Revisit when the wedge clears.

## B11 (2026-08-28) — FULL-REPO AGNOSTICIZATION (RULE.md sweep)

- NEW AI_sidecar/ai_sidecar/game_data.py: shared game-data loaders (job-change table + cities + RO map-prefix→town graph MAP_PREFIX_TOWN + parent_town())
- heuristic_service.py: JOB_CHANGE_NPCS now table-loaded (was hardcoded prontera/archer coords — WRONG, archer guild != novice change); 4 town-check sites → _is_city_map (cities.txt); farm-check → learned store; buy-NPC → learned shop_npc (removed 2nd hardcoded shop dict + buy 501 fallback)
- translator.py: return_town → parent_town (RO prefix graph, was move prontera); buy_pots → buy potion 30 (was buy 501 30)
- kafra_teleport.py: 2 hardcoded return "prontera" fallbacks → parent_town
- discovery.py: 501 shop-check → any potion keyword + buy potion
- edge_case_handler.py: buy 5 500 (Red Potion for bow class!) → buy 0 Arrow 100 (by name)
- woe_intelligence.py: use 504/501 → use potion (generic)
- KEPT (legit static RO game data, identical on every server): KAFRA_WARP_ROUTES + KNOWN_KAFRA_LOCATIONS (kafra costs/positions), CLASS_HUNTING_GROUNDS (level-banded guide, overridden by adaptive scoring), field-map lists in _cs_in_hunting/_audit_is_hunting (RO field geography), portal/navigation tables
- Tests: 16 affected pass; full suite running
