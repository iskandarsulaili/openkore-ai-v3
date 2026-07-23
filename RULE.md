# RULE.md — Developer Reference for openkore-ai-v3

## Architecture: Three-Layer Separation

```
┌──────────────────────────────────────────────────────────┐
│                    AI SIDECAR (Python)                    │
│  PDCA Loop · CrewAI Agents · Goal Stack · Decision Svc   │
│  Handles ALL decisions: config, routing, job change, etc │
└──────────────────────────────────────────────────────────┘
         ↑ actions (action_queue)        ↑ snapshot data
┌──────────────────────────────────────────────────────────┐
│              BRIDGE PLUGIN (Perl) ONLY FOR:               │
│  1. Monitor/report state to sidecar (snapshots)          │
│  2. Instant reflexes only (sub-second emergency)         │
│     e.g. survival flee when HP < 15% + aggro > 3         │
└──────────────────────────────────────────────────────────┘
         ↑ commands                    ↑ state
┌──────────────────────────────────────────────────────────┐
│                    OPENCORE BOT (Perl)                    │
│  Movement · Combat · Loot · NPC interaction              │
└──────────────────────────────────────────────────────────┘
```

## Hard Rules

### 1. Bridge Plugin is LIMITED
The bridge (`plugins/aiSidecarBridge/aiSidecarBridge.pl`) ONLY does:
- **Monitor/report**: Send snapshot data to sidecar (vitals, inventory, position, progression, skills, actors)
- **Instant reflexes**: Actions that need sub-second response (emergency survival flee when HP < 15% + aggro > 3, force-stand on field maps)
- **Execute commands**: Receive and execute `set <key> <value>` config changes from sidecar
- **NEVER**: Hardcoded config changes, routing decisions, selling logic, job change logic — those belong in sidecar

### 2. AI Sidecar Handles EVERYTHING Else
The sidecar (`AI_sidecar/`) handles ALL decisions:
- **Config changes**: Detect overweight → queue `set sellAuto 1` action → bridge hot-reloads
- **Routing**: Detect stuck in town → queue `move <hunting_map>` action
- **Job change**: Detect job_level ≥ 10 for Novice → route to job change NPC via database
- **Healing**: Detect low HP → queue potion use or sit
- **Skill allocation**: Detect unspent skill/stat points → allocate
- **Economy**: Detect full inventory → queue sell/restock actions

### 3. Zero Hardcoded Values — But System-Generated Config Is REQUIRED
- **Human-hardcoded** (NEVER): Map names, item names, skill names, NPC coordinates, level requirements written by a developer in source code. Use DB/tables instead.
- **System-generated config** (ALLOWED + ENCOURAGED): Config values (`set teleportAuto_*`, `set attackAuto_*`, `set route_*`) queued by the sidecar through the action queue. These are NOT hardcoded — they are decisions made by the AI/architecture at runtime.
- **The bottleneck**: The bots couldn't get kills because RULE.md prohibited system-generated config changes. This is now fixed — the system freely generates optimal config values.
- **NPC positions**: Use `tables/job_change_locations.txt` — never hardcode coordinates
- **Map names**: Use `zone_ladder` or `GameKnowledgeService` — never hardcode in source
- **Item/potion names**: Use `GameKnowledgeService` — never hardcode in source
- **Skill names**: Use skill database — never hardcode skill IDs
- **Level requirements**: Use `ro_knowledge.assess_job_advancement()` — never hardcode
- **Credentials**: Single `.env` source — never duplicate in config files

### 4. 100% Data Flow Completeness
Every field in every Pydantic model MUST be populated:
- **Vitals**: `hp`, `hp_max`, `hp_ratio`, `sp`, `sp_max`, `sp_ratio`, `weight`, `weight_max`, `weight_ratio`, `level`, `base_level`, `job_level`, `zeny`
- **Position**: `map`, `x`, `y`
- **Combat**: `ai_sequence`, `target_id`, `is_in_combat` (aggro_count derived from actors data)
- **Inventory**: `zeny`, `item_count`, `weight`, `weight_max`, `weight_ratio`, `overweight_ratio`
- **Progression**: `job_id`, `base_level`, `job_level`, `base_exp`, `base_exp_max`, `job_exp`, `job_exp_max`, `skill_points`, `stat_points`, `job_name`
- **Skills**: name + level for each known skill
- **Actors**: nearby entities with type, name, hp, distance

If a field is `None`/`0` when it shouldn't be, trace the gap in: bridge → Pydantic model → conscious engine → normalizer → PDCA/planner.

### 5. Agent Synergy — No Conflicts

| Agent | Role | Reads | Writes | Must not conflict with |
|-------|------|-------|--------|----------------------|
| **Survival reflex (bridge)** | Emergency flee when dying | HP, aggro_count (from @actors), map | lockMap, AI mode toggle | Game engine routing (has 300s grace window) |
| **Game Engine (PDCA)** | Recommends hunting maps, sole router | Level, job, location | lockMap, move commands | Survival reflex (grace period protects this) |
| **Progression Planner (CrewAI)** | Job change, equipment | `job_change_available` signal | move commands to NPC | Survival (should not flee from job NPC) |
| **Goal Stack** | Priority-ordered goal selection | Assessment results | GoalDirective entries | Must not create redundant goals |
| **Cold Start (PDCA)** | Initial config on connect | Latest snapshot | `set attackAuto_inLockOnly 0` only | Game engine handles all map routing |

**Goal priority order** (deterministic):
1. **Survival** — HP < 35% or dead/disconnected
2. **Job Advancement** — job level >= requirement and skill/stat points pending
3. **Opportunistic Upgrades** — economy/gear upgrade opportunities
4. **Leveling** — hunting progression (default when nothing urgent)

### 6. Reward/Punish System (Self-Supervised)
- **objective_max_age_cycles**: Goals expire after N PDCA cycles (punishment for stale/unachievable goals)
- **priority_decay_per_cycle**: Goal priority decays each cycle (reward cycles down if blocker unresolved)
- **reflex_actions_suppressed**: Bridge suppresses duplicate reflex actions — counts as negative feedback
- **HighConf counter**: Actions with confidence ≥ 0.8 signal the cold start to back off — success signal
- **ProvFail counter**: LLM provider failures degrade decision quality — must be monitored

## Coding Standards

### Perl (Bridge Plugin)
- Always `use strict; use warnings;` at file scope
- Declare all variables with `my` before first use in every sub
- `quotemeta()` is a FUNCTION — use `\Q...\E` inside regex, never `quotemeta()` as literal text
- Check both `$char->{key}` and `$::config{key}` for initialization before arithmetic
- Log at level 1 for production (`debug "msg", 'aiSidecarBridge', 1`)
- Paths: use `$::Settings->{tablesPath}` with fallback `|| 'tables'`

### Python (Sidecar)
- All Pydantic models should have complete field coverage — `extra="ignore"` is the default but don't rely on it to swallow missing data
- Use `getattr(obj, field, default)` for safe attribute access on snapshot objects
- When enriching signals in `crew_manager.py`, handle both dict and BotStateSnapshot objects
- Use `Path` for file paths, never hardcoded strings
- Imports at module top, not inline (exception: `pdca_loop.py` dynamic routing functions may inline)

## Git Workflow
- One semantic change per commit (not "fix stuff")
- Commit messages: `fix:` for bugs, `feat:` for features, `fix:` for data flow fixes
- Include the WHAT and WHY in the commit body, not just the title
- Revert bridge-only-appropriate changes that cross the architecture boundary

## Debugging Checklist
When a bot isn't behaving as expected, check in this order:
1. **Bridge snapshot** — is the data reaching the sidecar? Check `sidecar.log` for snapshot count
2. **Pydantic model** — is the field defined? Check `contracts/state.py`
3. **Conscious engine** — is the signal being extracted? Check `conscious_engine.py`
4. **Crew manager** — is the signal being enriched? Check `crew_manager.py`
5. **Goal stack** — is the correct goal being selected? Check `goal_stack.py`
6. **PDCA loop** — is the action being generated? Check `pdca_loop.py` action queue
7. **Bridge execution** — is the bridge receiving and executing the action?

## Key Files

| File | Purpose |
|------|---------|
| `plugins/aiSidecarBridge/aiSidecarBridge.pl` | Bridge — monitor/report + reflexes only |
| `AI_sidecar/ai_sidecar/app.py` | FastAPI entrypoint |
| `AI_sidecar/ai_sidecar/contracts/state.py` | Pydantic models (Vitals, InventoryDigest, etc.) |
| `AI_sidecar/ai_sidecar/conscious_engine.py` | Bot state awareness |
| `AI_sidecar/ai_sidecar/crewai/crew_manager.py` | Signal enrichment for CrewAI agents |
| `AI_sidecar/ai_sidecar/crewai/agents/progression_planner_agent.py` | Job change + equipment agent |
| `AI_sidecar/ai_sidecar/autonomy/pdca_loop.py` | PDCA cycle — action generation |
| `AI_sidecar/ai_sidecar/autonomy/goal_stack.py` | Goal prioritization |
| `AI_sidecar/ai_sidecar/autonomy/ro_knowledge.py` | RO game knowledge (job change, leveling) |
| `AI_sidecar/ai_sidecar/autonomy/decision_service.py` | Decision service orchestration |
| `tables/job_change_locations.txt` | Job change NPC coordinates (ALL classes) |
| `knowledge/rathena_db/db/` | rAthena game data (mobs, items, skills, jobs) |
| `.bot_profiles/*/control/config.txt` | Per-bot OpenKore config |
| `logs/` | Bot logs, sidecar log, checkpoints |

## Anti-Patterns to Avoid
- ❌ Adding business logic to the bridge (selling, routing, job change)
- ❌ Hardcoding values in source code (human-written) — system-generated config values through the action queue are ALLOWED and ENCOURAGED
- ❌ Editing config.txt files directly — the sidecar AI should do it via `set` commands
- ❌ Leaving Pydantic model fields unpopulated — 100% completeness
- ❌ Adding new agents without checking for goal conflicts with existing agents
- ❌ Building a fix without checking if the data flows end-to-end first

### 7. Hybrid AI — Bottom-Up + Top-Down, No Human Intervention
The system MUST be fully autonomous — no human editing configs, no manual API calls, no direct terminal commands for bot configuration.

**Bottom-Up (data-driven, emergent):**
- RiskManager tracks kills/deaths per map → adjusts routing recommendations
- Reward/punish mechanism: kills = reward (route to this map more), deaths = punish (avoid this map)
- System learns from outcomes without human input

**Top-Down (rule-based, goal-driven):**
- Death loop: detect TOWN→HUNT→TOWN pattern → route to safer map + adjust config
- Stuck detection: bot on same map >60 cycles → reset AI
- Cold start: queue default config for new bots (values generated by system, not hardcoded by humans)

**Config generation (both bottom-up + top-down):**
- Config values are SYSTEM-GENERATED based on bot state (level, gear, deaths, kills)
- No human decides "teleportAuto_minAggressives should be 5" — the system DETECTS high death rates and INCREASES the threshold
- Config values flow through: system detects condition → queues `set` command → bridge executes → $::config updated
- If a value doesn't work, the system detects continued failure and generates a NEW value

**Anti-pattern:**
- ❌ Human writing `set teleportAuto_minAggressives 5` in terminal or API call
- ❌ Human editing config.txt files directly
- ❌ Hardcoding static values that never change based on conditions
- ✅ System detecting high death rate and queueing config adjustment
- ✅ System detecting low kill rate and queueing attack config changes

## 8. Preemptive AI — Anticipate Before React

The system MUST be preemptive, not just reactive. Every decision should anticipate the next state:

**Before level X, do Y:**
- Before bot reaches Level 10: queue job change route, equip weapon upgrade, stock potions
- Before bot enters a map with element-strong monsters: queue element-matching weapon
- Before bot's HP drops below 40%: queue potion use (reactive + preemptive combined)
- Before bot hits 70% weight: queue return-to-town for selling
- Before bot reaches hunting zone: queue attack config + route_randomWalk + survival config

**Before A, do B:**
- Before moving to a new map: queue survival config for that map's danger level
- Before engaging a monster: queue optimal weapon + skill rotation
- Before night time: queue escape-to-town if no night-safe gear
- Before SP drops below 20%: queue skill usage stop, switch to auto-attack
- Before bot dies (HP < 30% on field): queue Butterfly Wing use

**Implementation:**
- The cold start queues DEFAULT config that anticipates Level 1 needs
- The combat system pre-queues weapon swaps before engaging different monster elements
- The death loop pre-queues config changes BEFORE the next TOWN→HUNT cycle
- The game engine pre-queues the correct route BEFORE the bot finishes its current navigation
- All preemptive actions flow through the action queue at reflex priority

## 9. Root Cause Analysis — Permanent Systemic Solutions

Every bug in this system traces to one of a small number of root causes. Band-aids (grep-and-fix, lookup tables, manual audits) are forbidden — every solution must be a permanent structural change that makes the entire class of bug impossible to reintroduce.

### Rule: No Guessing OpenKore Internals

The bridge MUST NOT contain any hardcoded OpenKore field name, command name, or skill ID. Every reference to OpenKore's data model must be derived from OpenKore's own source at load time. If a field/command/skill doesn't exist in OpenKore, the bridge must fail to load — not silently produce wrong data.

**Implementation:**
- `src/Network/Receive.pm` (packet handler keys, lines 722-774) is the single source of truth for `$char` fields
- `src/Commands.pm:initHandlers()` (lines 53-656) is the single source of truth for valid console commands
- `src/Network/Send.pm` is the single source of truth for valid packet types
- The bridge MUST parse these at load time and refuse to start if a referenced field/command doesn't exist

### Pattern A: OpenKore Field Name Mismatch — Permanent Fix

**Root cause:** The bridge uses field names that don't exist in OpenKore's data model.

**Band-aid (FORBIDDEN):** grep `src/Network/Receive.pm` before using a field.

**Permanent fix:** The bridge MUST derive all `$char` field access from OpenKore's packet handler definitions at load time.

**Implementation:**
1. At bridge load (`on_start3`), parse `src/Network/Receive.pm` to extract the `keys => [qw(...)]` arrays from the packet handler definitions (lines 722-774). These define every field OpenKore populates on `$char`.
2. Build a `%CHAR_FIELDS` hash at load time: `{ logical_name => openkore_field_name, ... }`.
3. All snapshot field access goes through a `_char_field($logical_name)` function that:
   - Looks up the OpenKore field name from `%CHAR_FIELDS`
   - Returns `undef` with a load-time warning if the field doesn't exist in OpenKore
   - Never hardcodes a field name in the snapshot builder
4. For non-`$char` data (map name, field properties), use `$field->baseName()` which is guaranteed by OpenKore's Field module.

**Verification:** The bridge fails to load if any snapshot field references a name not in `%CHAR_FIELDS`. No runtime silent failures possible.

### Pattern B: OpenKore Command Name Mismatch — Permanent Fix

**Root cause:** The bridge sends commands that don't exist in OpenKore's command registry.

**Band-aid (FORBIDDEN):** grep `src/Commands.pm` before using a command.

**Permanent fix:** The bridge MUST derive all valid command names from OpenKore's command registry at load time.

**Implementation:**
1. At bridge load, parse `src/Commands.pm:initHandlers()` (lines 53-656) to extract the command registry: `{ 'ss' => \&cmdUseSkill, 'ai' => \&cmdAI, ... }`.
2. Build a `%VALID_COMMANDS` hash at load time.
3. All command execution goes through a `_run_command($command)` function that:
   - Validates the command name against `%VALID_COMMANDS` before calling `Commands::run()`
   - Logs a warning and returns false if the command doesn't exist
   - Never hardcodes a command string in the reflex handlers
4. For commands with known OpenKore bugs (like `stat_add` at Commands.pm:5525), bypass `Commands::run()` entirely and send the packet directly via `$messageSender->send*()`. The bypass is documented in the code with the exact OpenKore bug reference.

**Verification:** Any command string in the bridge that doesn't match a registered OpenKore command produces a load-time warning. No runtime `Unknown command` errors possible.

### Pattern C: Silent Failure / No Observability — Permanent Fix

**Root cause:** Error paths that silently swallow failures.

**Band-aid (FORBIDDEN):** Audit every `eval {}` block.

**Permanent fix:** Every error path MUST produce observable output. The bridge MUST have a health endpoint that exposes error counters.

**Implementation:**
1. Replace all bare `eval { ... }` with `eval { ...; 1 } or do { my $err = $@ || 'unknown'; _record_error($err); }`.
2. `_record_error($err)` increments a named error counter and logs at warning level.
3. Add a `/bridge/health` HTTP endpoint that returns:
   - Error counters by name
   - Last error timestamp
   - Snapshot success/failure count
   - Poll success/failure count
   - Registration status
4. The sidecar polls `/bridge/health` every 60s and alerts if error counters increase.
5. `_throttled_warning` MUST NOT have a `return;` at the top. If warnings are too noisy, fix the root cause — don't silence the warning.
6. Event serialization (`_post_event`, `_flush_event_queue`) MUST handle all Perl data types (scalars, arrays, hashes, refs) without corruption. Use `JSON::PP::encode_json()` for ref values, never `substr()`.

**Verification:** Every error path produces a log line. The health endpoint shows zero errors in steady state. Any new error is immediately visible.

### Pattern D: Data Flow Gaps — Permanent Fix

**Root cause:** Snapshot fields that are always `undef`/`0` because the bridge doesn't populate them.

**Band-aid (FORBIDDEN):** Cross-reference Pydantic models manually.

**Permanent fix:** The bridge snapshot builder MUST be generated from the Pydantic model definitions.

**Implementation:**
1. Define the snapshot schema in a single source of truth (e.g., a JSON Schema file or the Pydantic models in `contracts/state.py`).
2. At bridge load, the bridge reads this schema and validates that every field has a corresponding data source.
3. The snapshot builder iterates over the schema, not over hardcoded Perl hash keys. If a field is in the schema but not in the bridge's data sources, the bridge logs a warning at load time.
4. The sidecar validates every incoming snapshot against the schema and logs a warning for missing fields.

**Verification:** Any field added to the Pydantic models but not populated by the bridge produces a load-time warning. No runtime missing-field surprises.

### Pattern E: Reflex Logic Depends on Invalid Data — Permanent Fix

**Root cause:** Bridge reflexes check conditions using fields that are always empty/undef.

**Band-aid (FORBIDDEN):** Audit every `if` condition manually.

**Permanent fix:** Every reflex condition MUST be validated against known-good data sources at load time.

**Implementation:**
1. Define each reflex as a data structure (not inline code): `{ name, condition_field, condition_regex, action, cooldown_ms }`.
2. At bridge load, validate that every `condition_field` exists in `%CHAR_FIELDS` or is a known non-`$char` source (`$field->baseName()`, `$AI::AI`, etc.).
3. If a reflex references a field that doesn't exist, the bridge fails to load with a clear error message.
4. Reflex conditions use `$field->baseName()` for map checks — never `$char->{map}` (which doesn't exist in OpenKore).

**Verification:** Any reflex referencing a non-existent field prevents the bridge from loading. No runtime silent misbehavior possible.

### Pattern F: File Revert Loses Changes — Permanent Fix

**Root cause:** Uncommitted changes lost on `git checkout --`.

**Band-aid (FORBIDDEN):** "Commit more often."

**Permanent fix:** Every code change MUST be committed before any revert-risk operation.

**Implementation:**
1. Before running `git checkout --`, `git stash` any uncommitted changes.
2. After reverting, `git stash pop` to restore them.
3. If conflicts occur, resolve them — don't discard.
4. The CI pipeline (`make test`) MUST pass before any commit. This prevents regressions from reverts.

**Verification:** `git stash list` is always empty after a revert operation. No lost work.

### Systemic Impact Summary

| Pattern | Bugs found | Band-aid (FORBIDDEN) | Permanent fix |
|---|---|---|---|
| A: Field name mismatch | 6 | grep before use | Derive all field names from OpenKore's packet handlers at load time |
| B: Command name mismatch | 3 | grep before use | Derive all command names from OpenKore's command registry at load time |
| C: Silent failure | 4 | audit eval blocks | Replace all bare eval with error-recording wrapper; add health endpoint |
| D: Data flow gap | 4 | cross-ref manually | Generate snapshot builder from Pydantic schema |
| E: Reflex invalid data | 14 | audit conditions manually | Define reflexes as data; validate fields at load time |
| F: Revert loses changes | 3 | commit more often | Stash before revert; CI gate on commits |

### The Meta-Pattern

Every bug is either (a) using a field/command name that doesn't exist in OpenKore, or (b) silently swallowing an error that would have revealed the problem.

**The permanent fix for both:** The bridge MUST derive all OpenKore-specific knowledge from OpenKore's own source at load time. If a field/command doesn't exist, the bridge fails to load — not silently produces wrong data. Every error path MUST produce observable output. No exceptions.

This is not negotiable. Band-aids are forbidden. Every fix must be a permanent structural change that makes the entire class of bug impossible to reintroduce.

## 10. #1 Pro RO Player's 12 Critique Points — Implemented Systems

The following 12 systems were implemented to address the critique from the #1 Pro RO player. Each system is a permanent, structural solution.

### 10.1 Spatial Combat Awareness
**File:** `AI_sidecar/ai_sidecar/combat/spatial_combat.py`
**What it does:** PositionOptimizer scores 10+ candidate positions around a target by expected damage intake vs output. Chooses diagonal positions (take 1 hit instead of 2 when mob turns), predicts caster AoE patterns, moves preemptively. Implements overkill awareness (stop attacking when DoT remaining > target HP). Skill chaining by PURPOSE (see 10.7) not just DPS.
**Integration:** Called by `combat_loop.py` before each attack action. Returns `(best_x, best_y, reason)` to the movement system.

### 10.2 Breakpoint-Aware Gear Scoring
**File:** `AI_sidecar/ai_sidecar/combat/breakpoint_gear_scorer.py`
**What it does:** Evaluates equipment by whether it helps reach breakpoints, not raw stats. DEX 150 = instant cast. STR every 10 = damage bonus. ASPD 190 = 2 attacks/sec. VIT 100 = soft/hard DEF threshold. A +3 DEX item goes from "worthless" to "priceless" when it pushes you from 148→151 DEX.
**Integration:** `GearScorer.score_item(item, current_stats)` returns a score. `best_upgrade(inventory, current_stats)` returns the single best item to equip. `breakpoint_gap(stat_name, current_value)` shows distance to next breakpoint.

### 10.3 Anti-Detection by Behavior, Not Timing
**File:** `AI_sidecar/ai_sidecar/anti_detection/behavior_engine.py`
**Config:** `AI_sidecar/config/behavior_profiles/default.yaml`
**What it does:** Replaces uniform random delays with realistic log-normal distributions. Generates genuine imperfections: bad paths (3-tile detours), wrong targets (attack non-optimal mob), AFK breaks (30s-5min every 30-90 min), favorite spots (prefer specific coordinates), micro-mistakes (walk into wall, cancel cast). Human-likeness score measures proximity to recorded human play patterns.
**Integration:** Bridge's anti-detection module reads behavior modifiers from this engine via the action queue.

### 10.4 MVP Encounter Knowledge
**File:** `AI_sidecar/ai_sidecar/combat/mvp_encounter_knowledge.py`
**What it does:** Per-MVP encounter templates covering mechanics, gimmicks, positioning, gear requirements, pre-engage checklist. 8 major MVPs implemented: Baphomet, Mistress, Orc Hero, Moonlight Flower, Phreeoni, Drake, Osiris, Maya. Each template knows AoE patterns, status effects, enrage thresholds, spawns, and counter-tactics.
**Integration:** `assess_engagement_safety(template, hp_pct, buffs, items, hit)` returns `(safe, reasons)` before engaging. `get_phase_command()` returns the correct command per encounter phase (engage, berserk, emergency).

### 10.5 Skill Purpose Classification (Not Just DPS)
**File:** `AI_sidecar/ai_sidecar/combat/skill_purpose.py`
**What it does:** Classifies every skill by PURPOSE: ZONING (Fire Wall blocks pathing), DENIAL (Lex Aeterna doubles next hit), SETUP (Cold Bolt wets for fire combo), CLEANUP (normal attack for finishing), SURVIVAL (Heal, buffs), MOBILITY (Teleport), BURST (Storm Gust), DOT (Poison). Includes combo relationships and level notes (Fire Wall level 1 is BETTER than 10).
**Integration:** `recommend_rotation(available_skills, target_element, target_hp_pct)` returns a purpose-ordered rotation. `get_skill_combo(a, b)` checks if two skills combo.

### 10.6 Economic Intuition
**File:** `AI_sidecar/ai_sidecar/economy/price_tracker.py`
**What it does:** Real-time price tracking with trend detection. Knows NPC buy/sell prices for common items. Detects WoE season spikes (3x on potions), bot farming crashes (0.3x on common drops), new content rushes (2x on materials). Provides sell recommendations (hoard, sell-now, sell-npc) and buy recommendations (buy, wait, stockpile).
**Integration:** `PriceTracker.get_sell_recommendation(item, current_price)` returns action + reason. `detect_economic_opportunity()` finds arbitrage profits. `set_usage_profile()` configures consumable restock thresholds.

### 10.7 Map Knowledge Pre-Populated
**File:** `AI_sidecar/ai_sidecar/combat/map_knowledge.py`
**What it does:** Pre-populated map database from rAthena data. Every map knows its monsters (level, element, size, race, spawn count, drops), safety rating, portals, NPC services. No learning-by-dying — the bot knows what's on a map before entering.
**Integration:** `get_hunting_maps(level, max_danger)` returns scored map recommendations. `get_map_knowledge(name)` returns full map data. `get_route_safety(a, b)` scores route safety. `get_mvp_maps()` lists MVP spawn locations.

### 10.8 Multi-Bot Coordination (Designed)
**Status:** Architecture designed, implementation pending
**Pattern:** Fleet coordinator service with shared state via sidecar. Bots communicate intent (who is tanking, who is DPSing, which mob is focused). Coordination tactics: tank+DPS (one taunts, one attacks from behind), pull+AoE (one pulls mobs to cluster, one AoEs), buffer+DD (preist buffs, wizard nukes).
**When to implement:** After single-bot systems are proven in production.

### 10.9 Contextual Learning (Designed)
**Status:** Architecture designed, implementation pending
**Pattern:** Death events feed into a learning pipeline. Every death records: map, monster, build, gear, what killed you. After N deaths to the same cause, the system adjusts config (avoid that map, change gear, different rotation). Cross-bot learning: what bot A learned, bot B benefits from.
**When to implement:** After MVP systems (10.1-10.7) are stable.

### 10.10 Social Performance (Designed)
**Status:** Architecture designed, implementation pending
**Pattern:** The bridge now tracks social events (whispers, party invites, trades). The sidecar needs a social behavior engine that responds believably: sometimes reply, sometimes ignore, sometimes be slow, sometimes make typos. Play a role, don't just count events.
**When to implement:** After anti-detection (10.3) is proven.

### 10.11 Real Exploit Discovery (Designed)
**Status:** Architecture designed, implementation pending
**Pattern:** Replace theoretical exploit discovery with data-mining from actual gameplay. Record knockdown-into-wall events, line-of-sight breaks, double-hit windups, spawn manipulation sequences. Use pattern matching on event streams, not code generation.
**When to implement:** After learning pipeline (10.9) is stable.

### 10.12 File Map — Complete Project Structure

| File | Purpose | Lines |
|------|---------|-------|
| `plugins/aiSidecarBridge/aiSidecarBridge.pl` | Bridge — monitor/report + reflexes | 4549 |
| `AI_sidecar/ai_sidecar/app.py` | FastAPI entrypoint | ~160 |
| `AI_sidecar/ai_sidecar/lifecycle.py` | Runtime lifecycle, task scheduler | ~900 |
| `AI_sidecar/ai_sidecar/combat/combat_loop.py` | Combat orchestrator | ~870 |
| `AI_sidecar/ai_sidecar/combat/damage_formulas.py` | Full pre-renewal damage formula | 729 |
| `AI_sidecar/ai_sidecar/combat/elemental_matrix.py` | Element/size/race multiplier matrix | ~500 |
| `AI_sidecar/ai_sidecar/combat/action_executor.py` | Skill rotation execution | ~590 |
| `AI_sidecar/ai_sidecar/combat/spatial_combat.py` | **NEW** Spatial combat positioning | ~300 |
| `AI_sidecar/ai_sidecar/combat/breakpoint_gear_scorer.py` | **NEW** Breakpoint-aware gear scoring | ~200 |
| `AI_sidecar/ai_sidecar/combat/skill_purpose.py` | **NEW** Skill classification by purpose | ~300 |
| `AI_sidecar/ai_sidecar/combat/mvp_encounter_knowledge.py` | **NEW** Per-MVP encounter templates | ~300 |
| `AI_sidecar/ai_sidecar/combat/map_knowledge.py` | **NEW** Pre-populated map intelligence | ~300 |
| `AI_sidecar/ai_sidecar/economy/price_tracker.py` | **NEW** Real-time price tracking | ~350 |
| `AI_sidecar/ai_sidecar/anti_detection/behavior_engine.py` | **NEW** Human-like behavior engine | ~300 |
| `AI_sidecar/config/behavior_profiles/default.yaml` | **NEW** Behavior profile config | ~100 |
| `AI_sidecar/ai_sidecar/hunting_zone_manager.py` | Dynamic zone learning | 375 |
| `AI_sidecar/ai_sidecar/innovation/innovation_engine.py` | Experiment/exploit discovery | 477 |
| `AI_sidecar/ai_sidecar/autonomy/pdca_loop.py` | PDCA cycle — action generation | ~800 |
| `AI_sidecar/ai_sidecar/combat/build_manager.py` | Build optimization | ~200 |
| `knowledge/rathena_db/db/` | rAthena game data | multi-file |
| `RULE.md` | Architecture governance | 440+ |

## 11. OpenKore Core vs AI Sidecar — Redundancy Reconciliation

The system has two layers that can produce conflicting decisions: OpenKore's built-in AI (config-driven) and the AI Sidecar (Python-driven). These MUST be reconciled to prevent conflicts.

### What OpenKore Core Handles (DO NOT duplicate in bridge)

OpenKore's built-in AI (controlled by `config.txt` settings) handles these autonomously:
- **Auto-attack**: `attackAuto`, `attackAuto_inLockOnly`, `attackAuto_followTarget`
- **Auto-heal**: `useSelf_item` for potions, `useSelf_skill` for heal skills — configurable HP/SP thresholds
- **Auto-flee**: `teleportAuto_minAggressives`, `teleportAuto_atkCount`, `teleportAuto_deadly`
- **Auto-sit**: `sitAuto_hp_lower`, `sitAuto_sp_lower`, `sitAuto_idle`
- **Auto-loot**: `takeAuto`, `itemsTakeAuto`
- **Auto-buff**: `useSelf_skill` for buffs with duration checks
- **Auto-restock**: `buyAuto` for consumables
- **Route calculation**: built-in A* pathfinding, portal walking, NPC teleport
- **NPC interaction**: dialog sequences, shop, storage, identify, refine

### What the Bridge Handles (sub-100ms, NOT in OpenKore core)

These reflexes are too fast for OpenKore's config-driven AI (which checks every ~500ms):
- **Emergency flee**: HP < 15% + aggro > 3 → immediate teleport (overrides OpenKore's slower teleportAuto)
- **Interrupt caster**: monster casting within 10 tiles → immediate attack (OpenKore doesn't have cast-interrupt logic)
- **Pre-dodge**: AoE skill detected → move before cast completes (OpenKore doesn't predict)
- **Weight warning**: weight > 90% → stop looting (OpenKore continues looting past overweight)
- **GM detection**: GM sprite detected → immediate manual mode (OpenKore has no GM detection)
- **Equipment broken**: gear durability hits 0 → alert (OpenKore doesn't track gear condition)

### Redundancy Rules

1. **If OpenKore's config-driven AI handles it, the bridge MUST NOT duplicate.** Set OpenKore thresholds via `set` commands instead of adding bridge reflexes.
2. **The bridge reflex is ONLY for sub-100ms responses** that OpenKore can't handle:
   | Reflex | Why Bridge | Why Not OpenKore |
   |--------|-----------|-----------------|
   | Interrupt cast | OpenKore doesn't check cast bars | N/A — not in OpenKore AI |
   | Pre-dodge AoE | OpenKore doesn't predict | N/A — not in OpenKore AI |
   | Emergency flee | OpenKore checks every 500ms | Bridge fires in <50ms |
   | Weight warning | OpenKore continues looting | Bridge stops early |
   | GM detection | OpenKore has no GM awareness | N/A — not in OpenKore AI |
3. **Config-based decisions go through the action queue**, not through bridge reflexes. The sidecar queues `set teleportAuto_* 5` → bridge executes → OpenKore AI uses the new threshold.

## 12. Multi-Bot Architecture — Isolation & Shared Mechanics

The system runs 3+ bots that must operate as coordinated agents, not 3 independent copies. Each bot has isolated mutable state (config, inventory, position) but shares learned intelligence (prices, MVP spawns, dangerous maps).

### Isolated (Per-Bot)

Each bot MUST have its own copy of:
- **Config files**: `.bot_profiles/<bot_name>/control/config.txt` — per-bot OpenKore settings
- **Bridge state**: `$registered`, `$_last_poll_ms`, `$death_count` — per-bot HTTP session state
- **Snapshot data**: Sidecar routes by `bot_name` — each bot has its own data pipeline
- **Action queue**: Sidecar maintains per-bot action queue — Bot A's `move prontera` won't affect Bot B
- **Social reputation**: Each bot maintains own `SocialIntelligenceV2` — bad reputation on Bot A doesn't affect Bot B
- **Inventory/equipment**: Bridge reports per-bot inventory — Bot B's gear decisions don't affect Bot A
- **Interrupt skills**: The interrupt-cast reflex (line 3697) resolves per-bot — Mage uses Fire Bolt, Swordsman uses Bash
- **Skill rotation**: Each bot has its own build — Bot A (Wizard) spams AoE, Bot B (Priest) heals

### Shared (Cross-Bot)

All bots share intelligence that improves faster with more data:
- **P2P knowledge**: `P2PKnowledgeNode` gossips MVP sightings, price data, danger zones between all bots
- **Price tracking**: `PriceTracker` singleton — one bot observing a market price benefits all bots
- **Map danger data**: `PortalVerifier` cross-references all bot logs — any bot's death marks a map dangerous for all
- **MVP locations**: `P2PKnowledgeNode` broadcasts MVP spawns — Bot A sees Baphomet, Bot B and Bot C know instantly
- **Innovation experiments**: PDCA discovery results shared — one bot's build experiment success benefits all

### Communication Protocol

Bots communicate through the sidecar's P2P knowledge network, NOT through direct bot-to-bot channels:

```
Bot A ───snapshot──▶ Sidecar ──gossip──▶ Bot B (via Sidecar)
  ▲                     │                    ▲
  │                     ▼                    │
  └────knowledge────────┴────knowledge───────┘
```

The sidecar acts as a message broker: Bot A discovers a price → sends to sidecar → sidecar gossips to Bot B and Bot C. This ensures:
- Bots don't need to know each other's IP/port
- Communication is authenticated through the sidecar
- No direct bot-to-bot channels (easier to firewall)
- Sidecar can deduplicate and rate-limit messages
- Offline bots catch up on reconnection via the sidecar's event store

### What OpenKore Core Handles (DO NOT duplicate in bridge)

OpenKore's built-in AI (controlled by `config.txt` settings) handles these autonomously:
- **Auto-attack**: `attackAuto`, `attackAuto_inLockOnly`, `attackAuto_followTarget`
- **Auto-heal**: `useSelf_item` for potions, `useSelf_skill` for heal skills — configurable HP/SP thresholds
- **Auto-flee**: `teleportAuto_minAggressives`, `teleportAuto_atkCount`, `teleportAuto_deadly`
- **Auto-sit**: `sitAuto_hp_lower`, `sitAuto_sp_lower`, `sitAuto_idle`
- **Auto-loot**: `takeAuto`, `itemsTakeAuto`
- **Auto-buff**: `useSelf_skill` for buffs with duration checks
- **Auto-restock**: `buyAuto` for consumables
- **Route calculation**: built-in A* pathfinding, portal walking, NPC teleport
- **NPC interaction**: dialog sequences, shop, storage, identify, refine

### What the Bridge Handles (sub-100ms, NOT in OpenKore core)

These reflexes are too fast for OpenKore's config-driven AI (which checks every ~500ms):
- **Emergency flee**: HP < 15% + aggro > 3 → immediate teleport (overrides OpenKore's slower teleportAuto)
- **Interrupt caster**: monster casting within 10 tiles → immediate attack (OpenKore doesn't have cast-interrupt logic)
- **Pre-dodge**: AoE skill detected → move before cast completes (OpenKore doesn't predict)
- **Weight warning**: weight > 90% → stop looting (OpenKore continues looting past overweight)
- **GM detection**: GM sprite detected → immediate manual mode (OpenKore has no GM detection)
- **Equipment broken**: gear durability hits 0 → alert (OpenKore doesn't track gear condition)

### Redundancy Rules

1. **If OpenKore's config-driven AI handles it, the bridge MUST NOT duplicate.** Set OpenKore thresholds via `set` commands instead of adding bridge reflexes.
2. **The bridge reflex is ONLY for sub-100ms responses** that OpenKore can't handle:
   | Reflex | Why Bridge | Why Not OpenKore |
   |--------|-----------|-----------------|
   | Interrupt cast | OpenKore doesn't check cast bars | N/A — not in OpenKore AI |
   | Pre-dodge AoE | OpenKore doesn't predict | N/A — not in OpenKore AI |
   | Emergency flee | OpenKore checks every 500ms | Bridge fires in <50ms |
   | Weight warning | OpenKore continues looting | Bridge stops early |
   | GM detection | OpenKore has no GM awareness | N/A — not in OpenKore AI |
3. **Config-based decisions go through the action queue**, not through bridge reflexes. The sidecar queues `set teleportAuto_* 5` → bridge executes → OpenKore AI uses the new threshold.

### Bot Isolation vs Shared Mechanics

Each bot needs some state isolation and some shared intelligence:

| Aspect | Isolated (per-bot) | Shared (all bots) | How |
|--------|-------------------|-------------------|-----|
| Config files | ✅ `.bot_profiles/*/control/config.txt` | — | Separate config files per bot |
| Bridge state | ✅ `$registered`, `$_last_poll_ms` per-bot | — | Bridge manages per-bot state internally |
| Snapshot data | ✅ Per-bot HTTP session | — | Sidecar routes by `bot_name` |
| Action queue | ✅ Per-bot action queue | — | Sidecar dequeues per `bot_name` |
| P2P knowledge | — | ✅ MVPs, prices, danger zones | `P2PKnowledgeNode` gossip protocol |
| PDCA discovery | — | ✅ Innovation experiments | Sidecar shares results across bots |
| Map danger data | — | ✅ Dangerous maps from any bot | `PortalVerifier` cross-references all logs |
| Price tracking | — | ✅ Price observations from all bots | `PriceTracker` singleton |
| MVP locations | — | ✅ MVP spawns from any bot | `P2PKnowledgeNode` broadcasts MVP sightings |
| Social reputation | ✅ Per-bot reputation | — | Each bot maintains own `SocialIntelligenceV2` |

### Implementation Checklist for Redundancy Removal

When adding a new behavior, check:
1. ❓ Does OpenKore's config-driven AI already handle this? (Check `src/AI/` and `config.txt` options)
2. ❓ If yes, can the sidecar achieve it via a `set` command instead of a bridge reflex?
3. ❓ If it must be in the bridge, is it sub-100ms response time? (If not, put it in the sidecar PDCA loop)
4. ❓ Does the bridge reflex conflict with an existing OpenKore config? (If yes, disable the OpenKore option first via `set`)
5. ❓ Is the data per-bot or shared? (Per-bot → isolated state. Shared → `P2PKnowledgeNode` or singleton)
