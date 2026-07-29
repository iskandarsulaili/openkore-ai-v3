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
- **Action**: Walk to Prontera portal (prt_fild05: 373, 205) — portal to town where sellAuto/storageAuto trigger naturally
- **Mechanism**: Commands only (`stand`, `move 373 205`, `ai auto`). Uses `AI::dequeue()` to clear conflicting AI states so move takes effect. NO config overrides, NO NPC interaction.
- **Why portal not Kafra**: OpenKore's AI state machine ignores `talknpc` from idle state. Walking to town lets OpenKore's built-in sellAuto/storageAuto handle recovery natively. The bridge must NOT issue NPC commands.
- **Priority**: Survival reflexes ALWAYS win over heuristic strategy. Full priority hierarchy:
  ```
  Survival (don't die) > Combat (kill things) > Hygiene (weight/inventory) > Economy (buy/sell) > Routing (where to go)
  ```
- **Cooldown**: 10 seconds between triggers to prevent spam while bot walks to portal.
- **MUST NOT**: Set configs, change lockMap, issue buy/sell/economy commands, or interact with NPCs directly.

### 2. REFLEXES Cannot Override lockMap (Critical)
Discovered through debugging: bridge reflexes that set `lockMap = prontera` override the heuristic's lockMap and create an endless "move prontera" loop. All 5 such reflexes have been disabled.
- The heuristic owns lockMap decisions
- The bridge enforces lockMap consistency (always set to hunting map)
- No bridge code may change lockMap to a town

### 3. attackAuto Is Level-Dependent — Controlled by Heuristic Config Audit
The heuristic sets attackAuto based on bot level:
- **Level 1-10 (Novice, no gear)**: attackAuto=2 (attack when idle, don't chase). Prevents suicide runs at Thief Bugs.
- **Level 10+ (has gear, skills)**: attackAuto=3 (aggressive, chase targets). Efficient farming.
- The bridge must NEVER set attackAuto.
- Heuristic config audit sets ALL critical configs every cycle.
- Bridge fallback profile only activates when heuristic hasn't set a value.
- No component may override heuristic's config once set (`_sidecar_set_` flag).
- Bridge's role is enforcement (lockMap consistency, spam prevention), NOT config control.

### 4. AI Sidecar Handles ALL Decisions
The sidecar (`AI_sidecar/`) handles:
- **Config changes**: Detect overweight → queue `set sellAuto 1` action
- **Routing**: Detect stuck in town → queue `move <hunting_map>` via portal coords
- **Job change**: Detect job_level >= 10 for Novice → route to job change NPC
- **Healing**: Detect low HP → queue potion use or sit
- **Skill/stat allocation**: Detect unspent points → queue stat_add/skills add
- **Skill rotation**: After job change, queue skill use commands (Bash, Double Attack, Heal) for efficient farming
- **Economy**: Detect full inventory → queue sell/restock actions
- **Map progression**: Level-based progression through 1-99 ladder
- **Party**: Leader creates party, others join via invite. partyAutoShare=1 for shared experience.
- **Death handling**: On death signal → queue walk back to hunting map, re-buy potions if needed

### 5. COLD_START Economy-First Sequence (Keep Starting Gear)
Fresh spawns follow: set lockMap → farm Porings for drops → sell drops (NOT starting gear) → buy potions from drop money → portal to hunting map
- **CRITICAL**: Keep the starting Knife. A Novice with bare hands does 1-5 damage. A Novice with a Knife does 15-25 damage. Selling the Knife for 50z is a net loss.
- Farm Porings on prt_fild05 for drops (Jellopy, Sticky Muffler, etc.) — these sell for zeny
- Buy Red Potions (item 501) from drop money, NOT from selling gear
- Uses GameKnowledgeDB for NPC/portal lookups (works in any town)

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

### 7. Map Progression Ladder (Safe First, Dungeons Later)
Bots progress through maps based on level. **Safe field maps first, dungeons at level 15+**:
```
Level 1-10:  prt_fild01-05 (Porings, Lunatics, Fabre — 0 damage monsters, safe for Novice)
Level 10-15: pay_fild01-11 (outside maps, safe for first-job classes)
Level 15-25: pay_dun00 (Payon Cave 1F — Skeletons, Zombies, undead — 3-5x density)
Level 25-35: pay_dun01 (Payon Cave 2F — Munak, Bongun, Ghoul)
Level 35-50: orcsdun01 (Orc Dungeon — Orc Warriors, Orc Archers)
Level 50-70: iz_dun00-03 (Byalan Dungeon — Marine Sphere, Kukre, Vadon)
Level 70-85: ein_dun00-02 (Culvert — high density, good drops)
Level 85-99: alde_dun00-04 (Clock Tower) or mag_dun01 (Magma Dungeon)
```
**Map choice rule**: Melee classes favor dungeons with high passive-mob spawn density (undead in Payon Cave, Orcs in Orc Dungeon). Ranged classes can use larger maps since they attack from distance. All classes benefit from dungeon density (3-5x more kills per hour). **Never send a level 1-10 Novice to a dungeon — they will die in one hit.**

### 8. Economy Loop
- sellAuto enabled with maxWeight=70-80% (triggers when inventory is actually full, not at 30%)
- buyAuto for Red Potions (itemID 501, max 30) — buy from drop money, NOT from selling gear
- On death: DEATH state sells items, returns to hunt
- In TOWN_STUCK: sells items before returning to hunt
- **Anti-kite**: attackAuto_maxDistance=20 prevents chasing monsters into packs. attackAuto_fleeToTarget=0 prevents running toward monsters that are too far.

### 9. Game Constants vs Strategy Values
- **Game constants** (ALLOWED to hardcode): NPC positions (Kafra 290,224), portal coords (Prontera 22,203), item IDs (Red Potion 501), skill IDs (Bash NV_BASH), class IDs. These are game data that don't change.
- **Strategy values** (NEVER hardcoded): Which map to farm, what level to job change, how many potions to buy, attack distance, stat allocation order. These must come from the heuristic/DB.
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

### 11. Per-Class Config Audit
The heuristic's config audit must be per-class, not one-size-fits-all:
- **Swordsman**: attackDistance=5 (melee), attackMaxDistance=20, teleportAuto_minAggressives=8
- **Thief**: attackDistance=3 (dagger, close range), attackMaxDistance=15, teleportAuto_minAggressives=6
- **Acolyte**: attackDistance=7 (holy light range), attackMaxDistance=25, teleportAuto_minAggressives=4 (squishy)
- **Archer**: attackDistance=10 (bow range), attackMaxDistance=30, teleportAuto_minAggressives=3 (very squishy)
- **Mage**: attackDistance=8 (bolt range), attackMaxDistance=25, teleportAuto_minAggressives=2 (extremely squishy)
- **Novice**: attackDistance=3, attackMaxDistance=15, teleportAuto_minAggressives=8 (safe)
- Class is determined from snapshot data (job name or job level)

### 12. Party Synergy Rules
- partyAuto=2 (auto-accept invites) for non-leader bots
- partyAutoShare=1 (shared experience) — critical for leveling efficiency
- Party leader is the highest-level bot (not first alphabetically)
- Party member range: bots should stay within 15 cells of each other for shared experience
- Party composition: 1 tank (Swordsman) + 1 healer (Acolyte) + 1 DPS (Thief/Archer/Mage) is ideal
- Party buffs: Acolyte should auto-buff party members with Blessing (AL_BLESSING) and Increase AGI (AL_INCAGI) when available

### 13. Death Handling
When a bot dies:
- Respawns in Prontera (default town)
- Heuristic detects death via HP=0 or map change to town
- Queue: sell items → buy potions → walk back to hunting map
- If died in a dungeon, walk to nearest town first, then portal to dungeon
- If died 3+ times in the same map in 1 hour, switch to a safer map
- Death counter resets on successful kill

### 14. Skill Rotation (Post-Job-Change)
After job change, the heuristic queues skill use commands:
- **Swordsman**: Bash (NV_BASH) on every attack — 100% hit rate, high damage
- **Thief**: Double Attack (NV_DOUBLE) — passive, auto-procs. No active skill needed.
- **Acolyte**: Heal (NV_HEAL) on self when HP<50%. Holy Light (NV_HOLYLIGHT) on undead.
- **Archer**: Arrow Shower (NV_ARROWSHOWER) on 3+ mobs. Improve Concentration (NV_IMPOSITIO) before fights.
- **Mage**: Fire Bolt (NV_FIREBOLT) on Earth-weak. Cold Bolt (NV_COLDBOLT) on Fire-weak. Fire Wall (NV_FIREWALL) for safety.
- Skills are queued as `skill use <skill_id> <target>` commands through the action queue.

### 15. Testing & Verification Required
Every change to the bridge, heuristic, or config audit must pass the test harness before deployment:
- Run `python3 test_harness.py` before committing
- Test harness checks: config audit values, bridge reflex correctness, attack block logic, pipeline flexibility, RULE.md compliance, config override count
- A failing test harness is a BLOCKER — no commit may fail the harness unless the test itself is wrong
- The test harness runs offline (no server connection needed) and verifies code logic only

### 16. Server Failure Handling
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

### 17. God-Tier AI Philosophy (No Mimicry)
This system is a God-Tier AI, NOT a human mimicry bot:
- **Deterministic optimal decisions**: Every action is the mathematically best choice for the given state. No randomized delays, no "human-like" imperfections.
- **Perfect execution**: When the optimal action is to attack a Poring, attack immediately — don't wait a random 300-800ms to look human.
- **No anti-detection**: Detection avoidance is a losing game. Instead, operate at speeds and efficiencies that no human can match. If a GM investigates, the bot's perfect execution is the point.
- **Speed is a feature**: The fastest kill rate, the shortest downtime, the most efficient routing. God-tier means optimal, not invisible.
- **However**: Respect server rules. Don't flood commands faster than the server can process them (use rate limiting, not randomized delays).

## Config Management
- **All config adjustments must go through the AI system (heuristic / Pro RO agent)** — never manually edit .bot_profiles/*/control/config.txt
- The heuristic uses `set <key> <value>` commands to change config at runtime
- Manual config edits are forbidden because they bypass the AI system's decision-making and create drift between expected and actual config state
- The bridge must NEVER override config — it only executes commands from the heuristic
