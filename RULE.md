# RULE.md — Developer Reference for openkore-ai-v3

## Architecture: Three-Layer Separation

```
┌──────────────────────────────────────────────────────────┐
│                    AI SIDECAR (Python)                    │
│  PDCA Loop · Heuristic Service · Pro RO Agent            │
│  Handles ALL decisions: config, routing, economy, etc    │
└──────────────────────────────────────────────────────────┘
         ↑ actions (action_queue)        ↑ snapshot data
┌──────────────────────────────────────────────────────────┐
│              BRIDGE PLUGIN (Perl) ONLY FOR:               │
│  1. Monitor/report state to sidecar (snapshots)          │
│  2. Pre-command interception (Commands::run/pre)         │
│     - Block "move prontera" on hunting maps              │
│     - Redirect "move prontera" to portal in town         │
│  3. Portal exit reflex (move from 367,205 to center)     │
│  4. Portal route redirect (move to portal every 30s)     │
│  5. Town mode: route_randomWalk=0 prevents spam          │
│  6. Hunting map guard: lockMap stays on current map      │
│  7. Execute commands from sidecar action queue           │
│  NOT: Routing decisions, selling logic, config overrides │
└──────────────────────────────────────────────────────────┘
         ↑ commands                    ↑ state
┌──────────────────────────────────────────────────────────┐
│                    OPENCORE BOT (Perl)                    │
│  Movement · Combat · Loot · NPC interaction              │
└──────────────────────────────────────────────────────────┘
```

## Hard Rules

### 1. Bridge is LIMITED to Monitor + Intercept + Reflex
The bridge (`plugins/aiSidecarBridge/aiSidecarBridge.pl`) ONLY does:
- **Monitor/report**: Send snapshot data to sidecar
- **Pre-command hook** (`Commands::run/pre`): The LAST LINE OF DEFENSE
  - Bot in Prontera: "move prontera" → "move 22 203" (portal redirect)
  - Bot on hunting map: "move prontera" → BLOCKED (hunting guard)
  - Bot in PvP/GvG: allow through (player needs to leave)
- **Attack interception**: Block attack commands for monsters with mon_control attack_auto <= 0 (ignored). Cannot block Porings (attack_auto=1) or any farm target.
- **Portal exit reflex**: Bot at prt_fild05 (367, 205) → move to center (200, 200)
- **Portal route redirect**: Bot in Prontera with lockMap set → issue "move 22 203" every 30s
- **Town mode**: route_randomWalk=0 in Prontera prevents "move prontera" spam from OpenKore AI
- **Config enforcement**: lockMap always set to hunting map, attackAuto never overridden
- **Execute commands**: Receive and execute `set <key> <value>` from sidecar
- **NEVER**: Hardcoded config overrides (attackAuto, attackDistance), routing decisions, economy logic
- **ALLOWED commands in reflexes**: `stand`, `move <x> <y>`, `ai auto`, `AI::dequeue()`, `AI::clear()` — commands only, no config overrides

### 1a. Emergency Survival Reflex (Priority Override)
The bridge has ONE additional reflex beyond portal handling: the **Emergency Survival Reflex**.
- **Trigger**: HP < 20% AND weight > 70% AND NOT in town
- **Action**: Walk to nearest Kafra (prt_fild05: 290,224) — free storage deposit to reduce weight
- **Mechanism**: Commands only (`stand`, `move 290 224`, `ai auto`). Uses `AI::dequeue()` to clear conflicting AI states so move takes effect. NO config overrides.
- **Priority**: Emergency reflex ALWAYS wins over heuristic strategy. Survival > economy > routing > combat.
- **Cooldown**: 10 seconds between triggers to prevent spam while bot walks to Kafra.
- **MUST NOT**: Set configs, change lockMap, issue buy/sell commands, or interact with NPCs directly. The heuristic handles recovery after the bot reaches safety.

### 2. REFLEXES Cannot Override lockMap (Critical)
Discovered through debugging: bridge reflexes that set `lockMap = prontera` override the heuristic's lockMap and create an endless "move prontera" loop. All 5 such reflexes have been disabled.
- The heuristic owns lockMap decisions
- The bridge enforces lockMap consistency (always set to hunting map)
- No bridge code may change lockMap to a town

### 3. attackAuto MUST Be 3 (Aggressive) — Exclusively via Heuristic Config Audit
The heuristic sets attackAuto=3, sitAuto_hp_lower=20, itemsTakeAuto=2, sellAuto=1, etc.
via the config audit in `_assess_impl`. The bridge must NEVER set these configs.
- Heuristic config audit sets ALL critical configs every cycle
- Bridge fallback profile only activates when heuristic hasn't set a value
- Fallback profile defaults now match heuristic: sitAuto_hp_lower=20, teleportAuto_hp=10
- No component may override heuristic's config once set (`_sidecar_set_` flag)
- Bridge's role is enforcement (lockMap consistency, spam prevention), NOT config control

### 4. AI Sidecar Handles ALL Decisions
The sidecar (`AI_sidecar/`) handles:
- **Config changes**: Detect overweight → queue `set sellAuto 1` action
- **Routing**: Detect stuck in town → queue `move <hunting_map>` via portal coords
- **Job change**: Detect job_level >= 10 for Novice → route to job change NPC
- **Healing**: Detect low HP → queue potion use or sit
- **Skill/stat allocation**: Detect unspent points → queue stat_add/skills add
- **Economy**: Detect full inventory → queue sell/restock actions
- **Map progression**: Level-based progression through 1-99 ladder
- **Party**: Leader creates party, others join via invite

### 5. COLD_START Economy-First Sequence
Fresh spawns follow: set lockMap → sell starting gear → buy 10 red potions → portal to hunting map
- Uses GameKnowledgeDB for NPC/portal lookups (works in any town)
- Does NOT skip economy phase (this was causing 0-kill sessions)

### 6. Pro RO Stat Builds (Class-Specific)
Stat points are allocated by tracking level changes:
- Heuristic tracks `_last_level[bot_id]` per bot
- On level-up detected: allocate 5 stat points in class-appropriate order
- **Archer**: DEX (50) > AGI (30) > LUK (20) — DEX for hit rate, AGI for ASPD, LUK for crits
- **Thief**: AGI (50) > DEX (20) > STR (20) — AGI for ASPD + Double Attack proc rate, DEX for hit
- **Acolyte**: INT (50) > DEX (20) > VIT (10) — INT for Heal damage (nukes undead), DEX for cast time
- **Swordsman**: STR (40) > VIT (30) > DEX (20) — Bash has 100% hit rate, STR first
- **Mage**: INT (50) > DEX (20) — INT for damage, DEX for cast time reduction
- NO dependency on `stat_points` signal (which may not propagate)

### 7. Map Progression Ladder (Dungeon-First)
Bots progress through maps based on level. **Dungeons are preferred over field maps** because they have 3-5x spawn density:
```
Level 1-10:  pay_dun00 (Payon Cave 1F — Skeletons, Zombies, undead)
Level 10-20: pay_dun01 (Payon Cave 2F — Munak, Bongun, Ghoul)
Level 20-35: gef_dun00 (Geffen Dungeon 1F — Drainliar, Creamy, Flora)
Level 35-50: orcsdun01 (Orc Dungeon — Orc Warriors, Orc Archers)
Level 50-70: iz_dun00-03 (Byalan Dungeon — Marine Sphere, Kukre, Vadon)
Level 70-85: ein_dun00-02 (Culvert — high density, good drops)
Level 85-99: alde_dun00-04 (Clock Tower) or mag_dun01 (Magma Dungeon)
```
**Map choice rule**: Melee classes favor dungeons with high passive-mob spawn density (undead in Payon Cave, Orcs in Orc Dungeon). Ranged classes can use larger maps since they attack from distance. All classes benefit from dungeon density (3-5x more kills per hour).

### 8. Economy Loop
- sellAuto enabled with maxWeight=30% (triggers early)
- buyAuto for Red Potions (itemID 501, max 30)
- On death: DEATH state sells items, returns to hunt
- In TOWN_STUCK: sells items before returning to hunt

### 9. Zero Hardcoded Values — But System-Generated Config Is REQUIRED
- **Human-hardcoded** (NEVER): Map names, item names, skill names, NPC coordinates, level requirements written by a developer in source code. Use DB/tables instead.
- **System-generated config** (ALLOWED + ENCOURAGED): Config values (`set teleportAuto_*`, `set attackAuto_*`, `set route_*`) queued by the sidecar through the action queue.
- **NPC positions**: Use GameKnowledgeDB — never hardcode coordinates
- **Map names**: Use progression ladder — never hardcode in source
- **Item/potion names**: Use GameKnowledgeDB — never hardcode skill IDs

### 10. Single Routing Authority
Only ONE system decides where the bot goes: the heuristic service.
- Heuristic sets lockMap
- Bridge enforces lockMap consistency
- Pro RO agent may RECOMMEND routes but must NOT override lockMap via direct commands
- No bridge reflex may change lockMap

### 11. Testing & Verification Required
Every change to the bridge, heuristic, or config audit must pass the test harness before deployment:
- Run `python3 test_harness.py` before committing
- Test harness checks: config audit values, bridge reflex correctness, attack block logic, pipeline flexibility, RULE.md compliance, config override count
- A failing test harness is a BLOCKER — no commit may fail the harness unless the test itself is wrong
- The test harness runs offline (no server connection needed) and verifies code logic only

### 12. Server Failure Handling
When the game server is unreachable or timing out:
- Bots auto-reconnect with exponential backoff (2s, 4s, 8s, 16s, max 30s)
- All in-memory state is preserved during reconnection attempts
- State persistence saves every 60s to survive process restarts
- After 5 consecutive reconnect failures, the sidecar enters "degraded mode":
  - PDCA loop continues running (processes snapshots from last known state)
  - Heuristic emits survival-mode configs to prepare for reconnection
  - No strategic decisions (routing, economy, job change)
  - When connection restores, exits degraded mode automatically
- The bridge never crashes or blocks the main loop during connection issues

## Config Management
- **All config adjustments must go through the AI system (heuristic / Pro RO agent)** — never manually edit .bot_profiles/*/control/config.txt
- The heuristic uses `set <key> <value>` commands to change config at runtime
- Manual config edits are forbidden because they bypass the AI system's decision-making and create drift between expected and actual config state
- The bridge must NEVER override config — it only executes commands from the heuristic
