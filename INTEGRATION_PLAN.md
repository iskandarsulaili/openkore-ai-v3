# INTEGRATION PLAN: openkore-ai-pro features into openkore-ai-v3
# Goal: Production-ready completeness. Zero stubs, zero placeholders, zero dead code.

## Guiding Principles
- Keep all v3 features/design/concept intact — add, never remove
- Each module must have real working implementation — no empty stubs
- Integrate with existing PDCA loop, not replace it
- Follow RULE.md architecture (Bridge = relay, Heuristic = decision)
- All features must work WITH the existing cold-start pipeline

## Phase 1: Architecture Foundation

### 1.1 State System Upgrade
**What Pro has:** 17 specialized state builders (character, inventory, map, party, guild, buffs, pets, homunculus, mercenary, mount, equipment, NPC dialogue, quests, market, environment, ground items, instances)
**What v3 has:** One flat snapshot with basic fields
**Plan:** 
- Create `state/` module with Pydantic models for each domain
- Build a `StateCollector` that aggregates all state components
- Each state component is an independent module with its own collector
- Keep existing snapshot as the base, extend with specialized state
- Integrate into existing bridge snapshot via extra fields

### 1.2 Module Split (heuristic_service.py)
**What Pro has:** 20+ independent subsystem directories each with 5-17 modules
**What v3 has:** Single 3718-line `heuristic_service.py` handling everything
**Plan:**
- Extract into `domains/` directory with one file per subsystem:
  - `domains/combat.py` — combat decisions, tactics, skill usage
  - `domains/economy.py` — buy/sell/storage decisions
  - `domains/routing.py` — map movement, portal navigation
  - `domains/social.py` — party, guild, chat
  - `domains/progression.py` — leveling, job change, stats
  - `domains/npc.py` — NPC interaction, dialogue
  - `domains/quests.py` — quest tracking and execution
  - `domains/equipment.py` — gear management
  - `domains/consumables.py` — potions, buffs, food
  - `domains/crafting.py` — alchemy, cooking, forging
  - `domains/companions.py` — pets, homunculus, mercenary
  - `domains/pvp.py` — PvP/WoE tactics
  - `domains/instances.py` — instance dungeons
  - `domains/environment.py` — day/night, weather awareness
  - `domains/mimicry.py` — human-like behavior
- Each domain has: `.md` docs referencing RO game data, clean `__init__.py` exports
- Domain instances are registered in a `DomainRegistry` and iterated for decisions
- Existing heuristic_config_audit stays as the config authority

### 1.3 IPC Bridge Upgrade
**What Pro has:** ZMQ + HTTP dual protocol, 17 state builders, circuit breaker, exponential backoff
**What v3 has:** ZMQ only, flat snapshot, no circuit breaker
**Plan:**
- Add HTTP fallback to existing bridge (ZMQ first, HTTP second)
- Extend bridge snapshot to include Pro's 17 state components
- Add circuit breaker (10 failures = manual reset)
- Add connection quality metrics
- Keep existing bridge functionality — extend, don't replace

## Phase 2: Combat & Jobs (Core Gameplay)

### 2.1 Combat Tactics Engine
**Pro files:** `combat/tactics/tank.py`, `melee_dps.py`, `ranged_dps.py`, `magic_dps.py`, `support.py`, `hybrid.py`
**Plan:**
- Create `domains/combat/tactics/` with 6 tactics modules
- Each tactics module has: `select_target()`, `select_skill()`, `evaluate_positioning()`
- Tank: threat management, aggro generation, party positioning
- Melee DPS: burst damage, positioning near target
- Ranged DPS: distance maintenance, line-of-sight
- Magic DPS: spell rotation, elemental advantage
- Support: buff timing, healing priority, debuff removal
- Hybrid: adaptive role switching based on party needs
- Register tactics by class (Swordsman=Tank, Thief=MeleeDPS, Archer=RangedDPS, etc.)
- Integrate with existing attackAuto and mon_control

### 2.2 Job-Specific Modules (45+ Jobs)
**Pro files:** `backend_engine/jobs/` — 47 job modules
**Plan:**
- Create `domains/combat/jobs/` with modules for all 45+ jobs
- Each job module defines:
  - `skill_rotation`: ordered list of skills to use in combat
  - `stat_build`: stat allocation priorities (per-level)
  - `gear_preferences`: weapon/armor types
  - `attack_range`: melee/ranged distance
  - `combat_tactics`: which tactics module to use
- Job modules register via a `JobRegistry` (lazy-loaded, one per bot)
- Integrate with existing stat build system (extend, don't replace)
- Add skill usage commands: `skill use <id> <target>` via action queue

### 2.3 Skill Rotation Engine
**Pro has:** Per-job skill priorities, AoE awareness, SP management
**Plan:**
- Create `domains/combat/skills.py` with skill registry
- Each skill has: ID, name, SP cost, range, cast time, cooldown, AoE flag
- Rotation engine: select best skill based on current context (target HP, SP, cooldown)
- SP management: don't use skills if SP < 20%, use basic attack instead
- Integration: skills queued as `skill use` commands through bridge's action execution

### 2.4 Target Selection System
**Pro has:** `combat/targeting.py` with scoring system
**Plan:**
- Create target scoring: HP%, distance, element advantage, danger level, loot value
- Monsters ignored by mon_control get score=0 (never targeted)
- Prioritize: low HP monsters (quick kill) > high value monsters (good drops) > aggressive monsters (danger)
- Integrate with existing mon_control and attack block

## Phase 3: World Interaction

### 3.1 NPC Interaction System
**Pro has:** `npc/` (13 modules) — dialogue parser, response selector, service manager
**Plan:**
- Create `domains/npc/` with modules:
  - `dialogue.py`: Parse NPC dialogue, track conversation state, select responses
  - `services.py`: Identify NPC type (merchant, quest, storage, etc.)
  - `shop.py`: Buy/sell via NPC shops
  - `storage.py`: Kafra storage interaction
  - `repair.py`: Equipment repair
- Replace all hardcoded `talknpc` calls with dialogue engine
- NPC positions from GameKnowledgeDB (existing system)

### 3.2 Quest Automation
**Pro has:** `quests/` (5 modules) — quest tracking, daily quests, achievement tracking
**Plan:**
- Create `domains/quests/` with modules:
  - `tracker.py`: Track active quests, progress, objectives
  - `automation.py`: Auto-accept/complete quests
  - `daily.py`: Daily quest rotation (Eden, Gramps, etc.)
  - `achievement.py`: Achievement tracking
- Quest data from GameKnowledgeDB
- Integration: quest actions queued through action queue

### 3.3 Equipment Management
**Pro has:** `equipment/` (4 modules) — evaluation, upgrades, optimization
**Plan:**
- Create `domains/equipment/` with:
  - `manager.py`: Track equipped items, inventory, upgrades
  - `optimizer.py`: Suggest gear upgrades based on level/class
  - `swapper.py`: Auto-swap weapons for elemental advantage
- Integrate with buyAuto for gear purchases
- Equipment state from bridge snapshot

### 3.4 Crafting System
**Pro has:** `crafting/` (7 modules) + `backend_engine/crafting/` (6 modules)
**Plan:**
- Create `domains/crafting/` with:
  - `alchemy.py`: Potion brewing
  - `cooking.py`: Food buffs
  - `forging.py`: Weapon/armor smithing
  - `enchanting.py`: Equipment enhancement
- Recipes from GameKnowledgeDB
- Crafting actions queued through action queue

### 3.5 Instance Dungeons
**Pro has:** `instances/` (7 modules) — definitions, state tracking, planning
**Plan:**
- Create `domains/instances/` with:
  - `registry.py`: Instance definitions (entrance, requirements, rewards)
  - `coordinator.py`: Enter → complete → exit lifecycle
  - `state.py`: Track current instance state
- Instance data from GameKnowledgeDB

### 3.6 Consumables & Buffs
**Pro has:** `consumables/` (5 modules) — buffs, food, recovery
**Plan:**
- Create `domains/consumables/` with:
  - `buffs.py`: Auto-buff management (Blessing, AGI Up, etc.)
  - `recovery.py`: Potion usage at HP/SP thresholds
  - `food.py`: Food buff rotation
- Integrate with existing buyAuto for restocking
- Buff commands queued as `skill use` actions

### 3.7 Companion Management
**Pro has:** `companions/` (5 modules) — pets, homunculus, mercenary, mount
**Plan:**
- Create `domains/companions/` with:
  - `pets.py`: Pet intimacy, feeding, evolution
  - `homunculus.py`: Homunculus stats, skills, AI
  - `mercenary.py`: Mercenary hiring, management
  - `mount.py`: Mount usage
- Companion state from bridge snapshot

## Phase 4: Intelligence & Adaptation

### 4.1 LLM Integration
**Pro has:** `llm/` (6 modules) + `backend_engine/llm/` (9 modules) — multi-provider
**Plan:**
- Create `llm/` with provider support:
  - `providers/openai.py`, `providers/azure.py`, `providers/deepseek.py`, `providers/anthropic.py`
  - `manager.py`: Provider selection, fallback chain, retry logic
  - `config.py`: Provider configuration from env/config
- Wire into existing `/v1/conscious/` endpoint
- LLM used for: NPC dialogue, quest decisions, strategic planning
- Keep existing rule-based heuristic as default; LLM as enhancement

### 4.2 Opponent Modeling
**Pro has:** `premium/opponent_modeling/` (5 modules)
**Plan:**
- Create `domains/combat/opponent_modeling.py`:
  - Track monster spawn patterns (time, location, density)
  - Predict monster behavior (aggro range, patrol paths)
  - Learn which monsters are dangerous vs farmable
  - Feed predictions into target selection and mon_control
- Start with simple statistical model, evolve to ML over time

### 4.3 Environment System
**Pro has:** `environment/` (6 modules) — time, weather, day/night
**Plan:**
- Create `domains/environment/` with:
  - `time.py`: Game time tracking, time-of-day effects
  - `weather.py`: Weather detection and effects
  - `map_metadata.py`: Map properties (dungeon, field, safe zone)
- Environment data from bridge snapshot (map name, server time)

### 4.4 Navigation System
**Pro has:** `navigation/` (4 modules) — portal DB, pathfinding
**Plan:**
- Create `domains/navigation/` with:
  - `portals.py`: Portal database (map connections)
  - `pathfinding.py`: Dijkstra shortest-path routing
  - `actions.py`: Convert paths to move commands
- Integrate with existing lockMap system
- Portal data from GameKnowledgeDB

## Phase 5: Social & PvP

### 5.1 Advanced Party System
**Pro has:** `social/` (10 modules) — party, guild, chat, MVP
**Plan:**
- Create `domains/social/` with:
  - `party.py`: Party coordination, role assignment, formation
  - `chat.py`: Chat processing, response templates
  - `mvp.py`: MVP tracking, spawn timing, party coordination
- Extend existing party system (keep, don't replace)

### 5.2 Swarm Intelligence
**Pro has:** `backend_engine/swarm/` (9 modules) — formations, consensus, tactics
**Plan:**
- Create `domains/social/swarm/` with:
  - `communication.py`: Bot-to-bot messaging through bridge
  - `formation.py`: Party formations (line, box, spread)
  - `consensus.py`: Group decision-making (where to hunt, when to retreat)
  - `tactics.py`: Group combat tactics (focus fire, spread, kite)
- Integrate with party system
- Swarm actions through action queue

### 5.3 PvP/WoE System
**Pro has:** `pvp/` (8 modules) — threat assessment, WoE management
**Plan:**
- Create `domains/pvp/` with:
  - `arenas.py`: PvP arena tactics
  - `woe.py`: War of Emporium participation
  - `battlegrounds.py`: Battleground automation
  - `threat.py`: Player threat assessment
- Only activates in PvP maps (detected from map name)

## Phase 6: Progression & Learning

### 6.1 Character Progression
**Pro has:** `progression/` (8 modules) — lifecycle, advancement
**Plan:**
- Create `domains/progression/` with:
  - `lifecycle.py`: State machine (NOVICE→FIRST_JOB→SECOND_JOB→TRANS→ENDGAME)
  - `advancement.py`: Job change automation
  - `stat_distribution.py`: Per-class stat allocation
- Integrate with existing pipeline (cold start → progression)
- Extend cold start to handle all stages (not just novice gear)

### 6.2 Adaptive Learning
**Pro has:** `learning/` (4 modules) — experience replay, strategy adaptation
**Plan:**
- Create `domains/learning/` with:
  - `experience.py`: Record outcomes (success, death, loot, exp)
  - `adaptation.py`: Adjust strategies based on success rate
  - `replay.py`: Reinforcement learning replay buffer
- Learn from: deaths (adjust tactics), loot rates (adjust maps), exp rates (adjust grind)

### 6.3 Planning Engine
**Pro has:** `planning/` — high-level goal planning
**Plan:**
- Create `domains/planning/` with:
  - `goals.py`: Goal hierarchy (level → job → gear → maps)
  - `scheduler.py`: Task scheduling (when to grind, quest, craft)
  - `optimizer.py`: Efficiency optimization (best exp/hour, zeny/hour)

## Phase 7: Human Mimicry & Anti-Detection

### 7.1 Behavior Mimicry
**Pro has:** `mimicry/` (7 modules) — timing, movement, patterns
**Plan:**
- Create `domains/mimicry/` with:
  - `timing.py`: Human-like delays (randomized, statistical distribution)
  - `movement.py`: Natural movement patterns (not straight lines)
  - `patterns.py`: Session patterns (gaming sessions, breaks)
  - `randomization.py`: Randomize all actions slightly to avoid detection
- Wrap all action queue commands with mimicry layer
- Human delay distribution: log-normal, mean=500ms, std=300ms

### 7.2 Anti-Detection
**Plan:**
- Add to mimicry module:
  - GM detection: check for GM characters in area, log out if detected
  - Report avoidance: randomize behavior when player count is high
  - Session length randomization: log out after random intervals (30-90 min)
  - Multi-account staggering: don't move all bots at same time

## Implementation Order

Phase 1 is MANDATORY before anything else — it sets up the infrastructure.
Phases 2-7 can be implemented in parallel using subagents.

### Week 1: Foundation
1. State system with all 17 collectors
2. Domain module structure with registry
3. IPC dual-protocol + circuit breaker
4. HTTP fallback

### Week 2: Combat & Jobs
1. 6 combat tactics with real implementations
2. Job registry with 45+ modules
3. Skill rotation engine
4. Target selection system

### Week 3: World Interaction
1. NPC dialogue system
2. Quest automation
3. Equipment management
4. Crafting system

### Week 4: Intelligence & Social
1. LLM integration
2. Swarm intelligence
3. Environment system
4. Navigation system

### Week 5: PvP, Progression, Mimicry
1. PvP/WoE
2. Character progression
3. Adaptive learning
4. Human mimicry + anti-detection

## File Structure (to be created)

```
openkore-ai-v3/
├── AI_sidecar/ai_sidecar/
│   ├── state/                    # NEW — state system
│   │   ├── __init__.py
│   │   ├── collector.py          # StateCollector aggregator
│   │   ├── character.py          # CharacterState
│   │   ├── inventory.py          # InventoryState
│   │   ├── map_state.py          # MapState
│   │   ├── party.py
│   │   ├── guild.py
│   │   ├── buffs.py
│   │   ├── pets.py
│   │   ├── equipment.py
│   │   ├── dialogue.py
│   │   ├── quests.py
│   │   ├── market.py
│   │   ├── environment.py
│   │   ├── instances.py
│   │   └── companions.py
│   ├── domains/                 # NEW — domain modules
│   │   ├── combat/
│   │   │   ├── __init__.py
│   │   │   ├── tactics.py       # Tactics dispatcher
│   │   │   ├── targeting.py     # Target scoring
│   │   │   ├── skills.py        # Skill registry + rotation
│   │   │   ├── opponent_modeling.py
│   │   │   ├── tactics/
│   │   │   │   ├── tank.py
│   │   │   │   ├── melee_dps.py
│   │   │   │   ├── ranged_dps.py
│   │   │   │   ├── magic_dps.py
│   │   │   │   ├── support.py
│   │   │   │   └── hybrid.py
│   │   │   └── jobs/
│   │   │       ├── registry.py  # 45+ job registrations
│   │   │       └── *.py         # swordsman.py, thief.py, ...
│   │   ├── economy/
│   │   │   ├── __init__.py
│   │   │   ├── buy.py
│   │   │   ├── sell.py
│   │   │   └── storage.py
│   │   ├── routing/
│   │   │   ├── __init__.py
│   │   │   ├── navigation.py
│   │   │   └── portals.py
│   │   ├── social/
│   │   │   ├── __init__.py
│   │   │   ├── party.py
│   │   │   ├── chat.py
│   │   │   ├── guild.py
│   │   │   ├── mvp.py
│   │   │   └── swarm/
│   │   │       ├── communication.py
│   │   │       ├── formation.py
│   │   │       ├── consensus.py
│   │   │       └── tactics.py
│   │   ├── progression/
│   │   │   ├── __init__.py
│   │   │   ├── lifecycle.py
│   │   │   └── advancement.py
│   │   ├── npc/
│   │   │   ├── __init__.py
│   │   │   ├── dialogue.py
│   │   │   ├── services.py
│   │   │   ├── shop.py
│   │   │   └── storage.py
│   │   ├── quests/
│   │   │   ├── __init__.py
│   │   │   ├── tracker.py
│   │   │   ├── automation.py
│   │   │   ├── daily.py
│   │   │   └── achievement.py
│   │   ├── equipment/
│   │   │   ├── __init__.py
│   │   │   ├── manager.py
│   │   │   ├── optimizer.py
│   │   │   └── swapper.py
│   │   ├── crafting/
│   │   │   ├── __init__.py
│   │   │   ├── alchemy.py
│   │   │   ├── cooking.py
│   │   │   ├── forging.py
│   │   │   └── enchanting.py
│   │   ├── instances/
│   │   │   ├── __init__.py
│   │   │   ├── registry.py
│   │   │   └── coordinator.py
│   │   ├── consumables/
│   │   │   ├── __init__.py
│   │   │   ├── buffs.py
│   │   │   └── recovery.py
│   │   ├── companions/
│   │   │   ├── __init__.py
│   │   │   ├── pets.py
│   │   │   ├── homunculus.py
│   │   │   └── mercenary.py
│   │   ├── pvp/
│   │   │   ├── __init__.py
│   │   │   ├── arenas.py
│   │   │   └── woe.py
│   │   ├── environment/
│   │   │   ├── __init__.py
│   │   │   └── time.py
│   │   ├── mimicry/
│   │   │   ├── __init__.py
│   │   │   ├── timing.py
│   │   │   ├── movement.py
│   │   │   └── anti_detection.py
│   │   └── learning/
│   │       ├── __init__.py
│   │       ├── experience.py
│   │       └── adaptation.py
│   ├── planning/
│   │   ├── __init__.py
│   │   ├── goals.py
│   │   └── scheduler.py
│   └── llm/
│       ├── __init__.py
│       ├── manager.py
│       ├── providers/
│       │   ├── openai.py
│       │   ├── azure.py
│       │   ├── deepseek.py
│       │   └── anthropic.py
│       └── config.py
│
├── plugins/aiSidecarBridge/
│   └── aiSidecarBridge.pl       # EXTEND: 17 state builders, HTTP fallback, circuit breaker
│
└── RULE.md                      # UPDATE with new subsystem rules
```

## Verification Criteria
- Every module has a `__init__.py` with `__all__` exports
- Every class has a docstring
- Every function has real implementation (no `pass`, no `raise NotImplementedError`)
- All modules import cleanly (`python3 -c "from ai_sidecar.domains.combat import *"`)
- Test harness passes (31/31 tests)
- Bridge compiles without errors
- Sidecar starts and accepts connections
