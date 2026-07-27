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
- **Portal exit reflex**: Bot at prt_fild05 (367, 205) → move to center (200, 200)
- **Portal route redirect**: Bot in Prontera with lockMap set → issue "move 22 203" every 30s
- **Town mode**: route_randomWalk=0 in Prontera prevents "move prontera" spam from OpenKore AI
- **Config enforcement**: lockMap always set to hunting map, attackAuto never overridden
- **Execute commands**: Receive and execute `set <key> <value>` from sidecar
- **NEVER**: Hardcoded config overrides (attackAuto, attackDistance), routing decisions, economy logic

### 2. REFLEXES Cannot Override lockMap (Critical)
Discovered through debugging: bridge reflexes that set `lockMap = prontera` override the heuristic's lockMap and create an endless "move prontera" loop. All 5 such reflexes have been disabled.
- The heuristic owns lockMap decisions
- The bridge enforces lockMap consistency (always set to hunting map)
- No bridge code may change lockMap to a town

### 3. attackAuto MUST Be 3 (Aggressive)
The Pro RO agent and health monitor previously set attackAuto=2 (passive) which prevented bots from killing. Both have been fixed to attackAuto=3.
- Heuristic sets attackAuto=3 every cycle
- Bridge must NOT override attackAuto
- No component may set attackAuto < 3

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

### 6. Level-Tracking Stat Allocation
Stat points are allocated by tracking level changes:
- Heuristic tracks `_last_level[bot_id]` per bot
- On level-up detected: allocate 5 stat points in class-appropriate order
- **DEX first for ALL classes** (hit rate is the #1 bottleneck at low levels)
- Archer: DEX > AGI > STR > VIT
- Thief: DEX > AGI > STR > VIT
- Acolyte: DEX > INT > VIT > STR
- Swordman: DEX > STR > VIT > AGI
- Mage: DEX > INT > VIT > STR
- NO dependency on `stat_points` signal (which may not propagate)

### 7. Map Progression Ladder
Bots progress through maps based on level:
```
Level 1-10:  prt_fild04 (starter field — porings, lunatics, fabres, picky)
Level 10-20: prt_fild05 (Porings, Lunatics, Fabres)
Level 20-30: pay_fild01 (Porings, Poporings, Lunatics — better density for melee)
Level 30-40: pay_fild03
Level 40-50: prt_fild08
Level 50-60: gef_fild01 (Geffen field)
Level 60-70: pay_fild01 (Payon field)
Level 70-80: mjolnir_04
Level 80-85: gef_fild02 (Geffen dungeon)
Level 85-99: gefen_fild01 (endgame field)
```
**Map choice rule**: Melee classes favor flat maps with high passive-mob spawn density (porings, lunatics, fabres). Ranged classes can use larger maps since they attack from distance.

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

## Config Management
- **All config adjustments must go through the AI system (heuristic / Pro RO agent)** — never manually edit .bot_profiles/*/control/config.txt
- The heuristic uses `set <key> <value>` commands to change config at runtime
- Manual config edits are forbidden because they bypass the AI system's decision-making and create drift between expected and actual config state
- The bridge must NEVER override config — it only executes commands from the heuristic
