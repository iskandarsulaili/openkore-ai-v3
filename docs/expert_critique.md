=============================================================================
PHASE 3 — EXPERT IN-GAME CRITIQUE
Persona: Skeptical Veteran Pro Ragnarok Online Player
"Been playing since 2002. Won multiple WoE tournaments. Know every stat,
skill, card, refine, and mechanic in this game better than I know my own
family. If your bot can't survive 1 hour in Prontera, I'm gonna tear it apart."
=============================================================================

─────────────────────────────────────────────────────────────────────────────
LENS 1: DIRECT CRITIQUE — What the observation log actually shows
─────────────────────────────────────────────────────────────────────────────

1. YOUR BOTS CAN'T STAY ALIVE IN A TOWN
─────────────────────────────────────────
   ┌────────────────────────────────────────────────────────────────────┐
   │ Three hours of running and your bots have the exact same HP they  │
   │ started with — 13/107, 27/107, 6/107. One of them is at SIX HP    │
   │ for the entire hour. SIX. A Poring sneezes and it's dead. In      │
   │ Prontera. The safest town in the game. A level 1 novice has more  │
   │ survival instinct.                                                 │
   └────────────────────────────────────────────────────────────────────┘

   Yesterday (what's outdated): Whoever wrote these macros thought
   "emergency_no_heal" and "emergency_move_prontera" would save the bot.
   Maybe that worked on some 2010 private server where a GM gave free
   Yggdrasil berries. Today: it's a frantic waving of hands while your
   character bleeds out.

   Current Gaps:
   - Your bots have ZERO HP recovery mechanism. No potions, no heal
     skills, no regenerative gear check, no sitting to regen. Even
     a naked Novice sidestepping a Poring knows to sit.
   - The "emergency" concept is a misnomer. It's not an emergency
     protocol — it's EVERY cycle. Everything is an emergency = nothing is.
   - Macro "reflex_survival_orange" doesn't exist. You declared a
     potion-drinking macro but never wrote it. That's like bringing an
     empty bottle to a bar fight — all intention, zero effect.

   What Should Happen (Short-term fix mentality):
   - CHECK YOUR INVENTORY for HP potions. IF HP < 30% → drink one.
   - IF no potions AND HP < 50% → SIT. Human players sit to regen.
   - IF HP < 10% → teleport OR warp to a healer NPC. Running in circles
     with "emergency_move_prontera" while at 6 HP is what a bot that's
     never watched a human play RO does.

   Medium-term: Map-specific HP thresholds. In a danger map, 70% might
   be emergency. In Prontera, 10% is barely worth mentioning. Your bot
   treats every map like a boss room.

   Long-term: Learn survivability patterns — when to tank, when to flee,
   when to use which potion grade. A level 1 character doesn't need
   an Yggdrasil berry. It needs a stack of Red Potions.

2. YOUR BOT SPAMS "AI MODE" FASTER THAN A MACRO WIZARD
─────────────────────────────────────────────────────────
   ┌────────────────────────────────────────────────────────────────────┐
   │ "AI set to auto mode" / "AI set to manual mode" / "AI is already  │
   │ set to auto mode" — these three lines repeat hundreds of times.    │
   │ It looks like your sidecar bridge is fighting OpenKore for control │
   │ of the AI, and neither one is winning. This is what I call "two    │
   │ drivers on the steering wheel" syndrome.                           │
   └────────────────────────────────────────────────────────────────────┘

   Yesterday: Someone wrote a bridge that toggles AI mode to "manual"
   when the sidecar wants control, then "auto" when it's done. It
   made sense on paper — like shift between macro mode and strategy mode.

   Current Gaps:
   - The sidecar sets manual mode, does nothing (0 LLM calls proven),
     then immediately sets auto mode back. The cycle repeats in SECONDS.
     You're burning CPU and log space on a pointless tug-of-war.
   - While the AI is in manual mode, the bot can't auto-attack monsters
     or respond to danger. So when the bridge disables auto, your bot
     becomes a sitting duck... that immediately re-enables auto anyway.
   - There's no debounce. No cooldown. No "if mode is already X, don't
     toggle it again in the next 30 seconds." It's a race condition
     between the bridge code and OpenKore's event loop.

   What Should Happen:
   - Decide: Is this an auto-pilot bot with LLM override, or an LLM bot
     that occasionally uses OpenKore macros? Pick one architecture.
   - If it hybrid: Mode switches should have a DEBOUNCE — minimum 5-10
     seconds between toggles. Track the last switch time.
   - If sidecar has nothing to do (which it doesn't — 0 LLM calls),
     DON'T TOUCH the AI mode at all. Leave it in auto.

3. YOUR PARTY SYSTEM DETECTS GHOSTS
─────────────────────────────────────────
   ┌────────────────────────────────────────────────────────────────────┐
   │ "party_low_hp (player=openkoreaihuman HP=0/1=0, dist=0)" — this   │
   │ player has 0 HP out of 1 max HP. They are at DISTANCE 0.          │
   │ They don't exist. They're in another dimension. Your bot is       │
   │ casting party heals at a ghost.                                   │
   └────────────────────────────────────────────────────────────────────┘

   Yesterday: Party-sharing macros written for 2020-era RO where your
   whole party stays on screen together. Makes sense for party play.

   Current Gaps:
   - The "party member" being detected has 0/1 HP — this isn't a low-HP
     player, it's either: a) an invalid/ghost party entry, b) someone
     in another map, or c) the party data structure isn't populated
     correctly. A real player never has 1 max HP.
   - The filter should be: IF max_hp < 100 THEN skip (it's not a real
     player state). AND IF map != current_map THEN skip.
   - Instead, it triggers party_low_hp → sidecar generates a reflex →
     the reflex fails → cycle repeats.

   What Should Happen:
   - Validate party member data before reacting: max_hp must be > 100,
     map must match, distance must be meaningful.
   - If a party member can't be validated in 3 consecutive cycles,
     consider them offline and stop checking until re-connection event.

4. YOUR SIDECAR IS REJECTING ITS OWN BOTS' DATA
───────────────────────────────────────────────────
   ┌────────────────────────────────────────────────────────────────────┐
   │ 1,423 HTTP 422 validation errors in one hour. Your bots send      │
   │ event data to the sidecar, and the sidecar says "I don't know     │
   │ what this is" — 1,423 times. That's like sending a text message   │
   │ and getting "ERROR: MESSAGE FORMAT NOT RECOGNIZED" back every     │
   │ single time. Even a broken clock is right twice a day.            │
   └────────────────────────────────────────────────────────────────────┘

   Yesterday: API schemas were designed for a data format that bots
   were supposed to send. Probably worked in unit tests.

   Current Gaps:
   - The bots are sending events in a format that the /v2/ingest/event
     endpoint rejects. This means the bridge code (Perl plugin) and
     the server code (Python FastAPI) disagree on the schema.
   - A 422 means the server understood the request but found semantic
     errors. So the endpoint is reachable, the format is just wrong.
   - This has been broken for the entire hour, and there's no error
     handling or fallback on the bot side.

   What Should Happen:
   - Fix the schema mismatch between aiSidecarBridge.pl and the FastAPI
     /v2/ingest/event endpoint. Print both schemas, diff them.
   - Add client-side validation: before sending, check the event payload
     format matches what the server expects.
   - Add retry with backoff: if 422, don't re-send the same bad data
     — log the payload and move on.

5. YOUR LLM DOESN'T EXIST
───────────────────────────
   ┌────────────────────────────────────────────────────────────────────┐
   │ ZERO. Not one. Zero LLM calls in one hour. Zero planner runs.    │
   │ Your "conscious PDCA loop" has the consciousness of a rock.      │
   │ The startup gate never opened — it spent the whole hour in       │
   │ "startup_gate_initializing" and "planner_stale". You have built  │
   │ a Ferrari engine (sidecar) that's not connected to any wheels.   │
   └────────────────────────────────────────────────────────────────────┘

   Yesterday: The architecture doc probably says "PDCA loop with LLM
   reasoning generates objectives and plans." On paper, it's a
   beautiful diagram — circles and arrows and "autonomous AI."

   Current Gaps:
   - The PDCA loop runs, it just does nothing. The planner never fires.
   - 0 LLM calls means either: a) no LLM provider is configured,
     b) the planner precondition never evaluates to true, c) the
     planner crashes silently before making a call, or d) the startup
     gate blocks all planning.
   - "startup_gate_initializing" — for 60 minutes.
   - "planner_stale" — the planner has never been updated, so it has
     no last-updated timestamp, so it's "stale" from birth.

   What Should Happen:
   - This is the most critical fix. Without LLM planning, the entire
     "AI" part of "openkore-ai-v3" is marketing copy. You have a
     macro bot with a broken bridge.
   - Fix the startup gate: make it initialize properly instead of
     hanging in "initializing" forever. Or lift the gate after a
     timeout and run anyway.
   - Verify the LLM provider config actually works: can the sidecar
     make a single test call to the LLM API?
   - If the planner dies on precondition, log WHY it didn't fire
     (don't just stay silent).

6. YOUR REFLEX SYSTEM IS A SCREEN DOOR ON A SUBMARINE
────────────────────────────────────────────────────────
   ┌────────────────────────────────────────────────────────────────────┐
   │ 25,154 reflex triggers → 926 actions emitted → many of those      │
   │ also fail. 96.3% of triggers are useless. Your reflex system      │
   │ detects everything and fixes nothing. It's a car alarm that       │
   │ goes off every time a leaf falls but stays silent during a theft. │
   └────────────────────────────────────────────────────────────────────┘

   Yesterday: The reflex system was designed to catch every possible
   in-game event and rank them by priority. In theory: comprehensive.

   Current Gaps:
   - Sensitivity is turned up to maximum. Every HP tick, every movement,
   every party event triggers a reflex. The system is drowning in its
   own noise.
   - Suppression rate (96.3%) suggests the system knows most triggers
   are noise, but still spends all its CPU processing them before
   suppressing them.
   - When an action IS emitted, it always fails ("all targets failed").

   What Should Happen:
   - Pre-filter at the bot level: don't send events that are obviously
     noise (same HP as 5 seconds ago, same position, etc.)
   - Priority threshold: "emergency" should mean actual emergency
     (HP < 10% AND being attacked), not "HP dropped from 107 to 106"
   - Fix the action emitter so emitted actions actually reach the bot.
     "all_targets_failed" means the action was generated but couldn't
     be delivered — likely a bot identifier or session mismatch.

7. YOUR BOTS HAVE NO SENSE OF PROGRESSION
───────────────────────────────────────────
   ┌────────────────────────────────────────────────────────────────────┐
   │ After 1 hour, your bots haven't gained a single level, killed a   │
   │ single monster, or earned a single zeny. They don't hunt. They    │
   │ don't grind. They just stand in Prontera and panic.               │
   └────────────────────────────────────────────────────────────────────┘

   Yesterday: Bot scripts were written assuming the sidecar would tell
   them when and what to hunt. "Goal decomposition → assigned objectives."

   Current Gaps:
   - No default grind behavior. When the sidecar fails to provide
     objectives (which it always does — 0 planner runs), the bot should
     fall back to a default: "go to a training map and auto-attack the
     weakest monster in range."
   - There's no "idle" directive. The bots are online but idle.
   - The economy loop never activates (no loot, no selling, no buying).

   What Should Happen:
   - Add a survival-first idle loop: if no sidecar plan exists, apply
     default plan: move to training area, auto-attack, pick up loot,
     return to town to sell when inventory full.
   - Death recovery: what happens when a bot dies? There should be a
     respawn → return to spot → resume loop.
   - Only engage sidecar for strategic decisions (map change, boss
     hunting, priority targeting), not for "stand there and don't die."

─────────────────────────────────────────────────────────────────────────────
LENS 2: ASPIRATIONAL CRITIQUE — What the #1 Top Pro RO Player Demands
─────────────────────────────────────────────────────────────────────────────

  "If I were to build an AI bot that could stand alongside the best
   human players, here's what it would need. These aren't nice-to-haves.
   These are minimum requirements for a 'Pro' label."

┌─────────────────────────────────────────────────────────────────────────┐
│ SHORT-TERM FIXES (Days)                                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│ (A) SURVIVABILITY BASELINE                                              │
│     - Inventory check: potions present? → use at thresholds             │
│     - Sitting: HP < 50% → sit until > 80% (standard RO regen)          │
│     - ESCAPE: HP < 10% → Fly Wing OR teleport to safe map              │
│     - Port to healer NPC: if no pots and HP critical → warp to          │
│       Prontera healer → buy pots → return                               │
│     - ANY bot that can't survive 5 minutes unattended is not a bot.     │
│       It's a corpse with a connection.                                  │
│                                                                         │
│ (B) SILENCE THE NOISE                                                   │
│     - Fix $max_len. One undefined variable. OpenKore's -w flag          │
│       suppresses warnings, or initialize the variable to 0.             │
│     - Debounce AI mode toggling: no switches within 10 seconds.         │
│     - Party validation: ignore entries with max_hp < 100 or wrong map.  │
│     - A clean log is a fast log. Your bots spend ~70% of their          │
│       mental CPU printing Perl warnings. That's 70% waste.              │
│                                                                         │
│ (C) IDLE KILL LOOP                                                      │
│     - Default behavior when no objective: auto-attack the nearest       │
│       monster within level range on the current map or a training map.  │
│     - Without this, your bot is just an expensive heartbeat.            │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│ MEDIUM-TERM IMPROVEMENTS (Weeks)                                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│ (D) REAL LLM PLANNING                                                   │
│     - Fix the startup gate so planner actually fires.                   │
│     - The LLM should generate context-aware objectives:                 │
│       "I'm a level 1 Novice in Prontera. I should:                      │
│        1. Buy Red Potions from the tool dealer                          │
│        2. Walk to Payon Cave 1F                                         │
│        3. Auto-attack Zombies and Pickup Dead Bat Wings                 │
│        4. Return to town to sell when weight > 80%"                     │
│     - Objective priority should adapt to state: low HP → recovery,      │
│       no pots → economy, good state → grind.                            │
│                                                                         │
│ (E) CONTEXT-AWARE AUTONOMY                                              │
│     - Server-agnostic: the bot should detect server rates (exp,         │
│       drop, mob spawn) and adapt. Don't hardcode for "my private        │
│       server."                                                          │
│     - Map awareness: danger maps require different HP thresholds,       │
│       different escape routes, different loot filters.                  │
│     - Class awareness: A Mage needs SP management. An Archer needs      │
│       arrow crafting. A Merchant needs vend/buy logic. One-size         │
│       bot fits nobody well.                                             │
│                                                                         │
│ (F) REFLEX INTELLIGENCE                                                 │
│     - Not all events are equal. Rate-limit triggers: "HP changed"       │
│       should trigger at most once per 2 seconds, not 50 times/sec.      │
│     - Suppression should be at source (bot plugin), not at destination  │
│       (sidecar). Don't send what you're going to suppress.              │
│     - Action delivery: fix "all_targets_failed" so emitted actions      │
│       actually reach the intended bot.                                  │
│                                                                         │
│ (G) DATABASE-FIRST CONFIG                                               │
│     - No hardcoded paths, thresholds, maps, or item IDs.                │
│     - All bot configuration in a queryable DB: preferred maps by        │
│       level bracket, potion thresholds by class, loot/no-loot lists.    │
│     - This enables server-agnostic adaptivity without code changes.     │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│ LONG-TERM VISION (Months)                                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│ (H) COMPETITIVE INTELLIGENCE                                            │
│     - WoE-ready: detect warps, flag carriers, EMP targets. Use          │
│       LLM to coordinate 3+ bots in real-time. One pushes, one           │
│       breaks, one snipes the emp.                                       │
│     - MVP-level PvP: timing of stuns, dispels, potion lock.             │
│       React to player behavior, not scripted sequences.                 │
│     - Economy arbitrage: scan market prices, identify flips,            │
│       buy low/sell high across multiple characters.                     │
│                                                                         │
│ (I) SELF-HEALING ARCHITECTURE                                           │
│     - If the sidecar crashes: bots fall back to fully autonomous        │
│       macro mode (no LLM dependency for basic survival).                │
│     - If the LLM provider fails: same fallback, with cached             │
│       last-good plan.                                                   │
│     - If a bot disconnects: auto-reconnect with same session.            │
│     - If a bot dies: respawn, analyze death cause, update strategy.     │
│     - The system should PROVE it can self-recover, not just claim it.   │
│                                                                         │
│ (J) TEAM PLAY                                                           │
│     - 3 bots, but zero teamplay observed. They operate on the same      │
│       server but don't coordinate. No shared party, no shared loot,     │
│       no buffs, no support chain.                                       │
│     - Multi-char coordination: one FS Priest buffing two DPS.           │
│       Auto-follow, auto-heal, share loot filter.                        │
│     - Cross-char economy: one Merchant vends what the others farm.      │
│                                                                         │
│ (K) PROOF, NOT PROMISES                                                 │
│     - Every claim should be verifiable. "PDCA running" for 1 hour       │
│       with 0 planner runs is a LIE exposed by telemetry.                │
│     - Health checks should check the ACTUAL THING, not report           │
│       "running" when the subsystem returns null for every metric.       │
│     - If the startup gate is stuck: ALARM. Don't silently stay          │
│       open for an hour.                                                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

─────────────────────────────────────────────────────────────────────────────
CLOSING VERDICT
─────────────────────────────────────────────────────────────────────────────

  Right now, openkore-ai-v3 is not an AI bot. It's a Perl macro setup
  with a broken Python sidecar that thinks really hard about doing
  something and then does nothing.

  The bones are there: 3 bots running stable for an hour is actually
  impressive for uptime. The sidecar stays up. The reflex system catches
  events. But nothing WORKS. It's a car with the engine running, the
  wheels spinning in the air, and the driver screaming "I'M DRIVING!"

  What I want to see:
  - Bots that can stay alive unattended (HP goes up, not down)
  - Bots that actually kill things and loot
  - An LLM that proves it's connected by generating ONE objective
  - A reflex system that fixes instead of just detecting
  - Logs that don't look like someone's cat walked on the keyboard

  Get the foundations right. Survival. Grind. Economy. Then we talk
  about WoE and PvP. Right now we're not even past tutorial island.

  — A Pro RO Player Who's Seen Too Many "AI Bots"
=============================================================================
