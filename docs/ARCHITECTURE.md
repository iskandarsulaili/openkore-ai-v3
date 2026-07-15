# openkore-ai-v3 Architecture Document

> **Version:** 2.0  
> **Date:** 2026-07-15  
> **Author:** Pro RO Player / AI Engineer  
> **Scope:** Full-stack architecture for solo and multi-bot RO automation  
> **Status:** All items marked "NOW" must be implemented. Only SaaS is "FUTURE."

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Core Architecture: Hybrid Bottom-Up + Top-Down](#2-core-architecture-hybrid-bottom-up--top-down)
3. [Layer 1: Bridge (Bottom-Up Reflex Layer)](#3-layer-1-bridge-bottom-up-reflex-layer)
4. [Layer 2: Sidecar (Middle Tactical Layer)](#4-layer-2-sidecar-middle-tactical-layer)
5. [Layer 3: LLM / Conscious Engine (Top-Down Strategic Layer)](#5-layer-3-llm--conscious-engine-top-down-strategic-layer)
6. [Transport: HTTP as Cross-Platform Backbone](#6-transport-http-as-cross-platform-backbone)
7. [Solo Gameplay Architecture](#7-solo-gameplay-architecture)
8. [Multi-Bot / Fleet Architecture](#8-multi-bot--fleet-architecture)
9. [Combat System Deep Dive](#9-combat-system-deep-dive)
10. [Economic System](#10-economic-system)
11. [Learning & Adaptation](#11-learning--adaptation)
12. [Observability & Safety](#12-observability--safety)
13. [Windows Compatibility](#13-windows-compatibility)
14. [SaaS Architecture (FUTURE ONLY)](#14-saas-architecture-future-only)
15. [Issue Registry: Every Problem Addressed](#15-issue-registry-every-problem-addressed)

---

## 1. System Overview

openkore-ai-v3 is a **hybrid AI system** for automating Ragnarok Online gameplay. It combines:

- **Bottom-up reflexes** (sub-100ms, hardcoded, no LLM) for combat survival
- **Middle-layer tactics** (pattern detection, rule adjustment, 500ms-2s) for adaptation
- **Top-down strategy** (LLM-driven, 30s-5min) for high-level planning

The system runs as a **fleet of bots** (1-N accounts) coordinated through a central sidecar.

### 1.1 Key Discovery: Bridge Runs Before AI Attack Loop

The bridge's `_check_bridge_reflexes()` fires in `mainLoop_pre`, which runs **before** `AI::Attack::process()` in `CoreLogic.pm`. This means:

- **Reflexes preempt attacks**: If HP < 50%, the heal reflex fires before the attack loop even starts
- **No race conditions**: The bridge can cancel an attack before it begins
- **Critical for survival**: Emergency reflexes (heal, flee, teleport) always win over combat

This is the correct architecture — the bridge acts as a **safety gate** that the attack AI must pass through.

### 1.2 Key Discovery: OpenKore's Built-in Combat is Config-Driven

`AI::Attack.pm` (1129 lines) uses `attackSkillSlot_*` config entries for skill selection. It iterates through numbered slots and checks conditions. This is **static** — the config is loaded at startup and doesn't change dynamically. Our sidecar overrides this by pushing new config values through the bridge's HTTP API, effectively hot-reloading the attack skill slots.

The attack engine has:
- **`process()`**: Dispatcher that validates target, checks for killsteal, handles approach
- **`main()`**: Core combat brain — predicts movement, chooses attack method (weapon/skill/combo), handles kiting via `runFromTarget`, anti-stuck logic
- **Combo system**: `attackComboSlot_*` for skill chains (e.g., Bash → Magnum Break)
- **Kiting**: Built-in `runFromTarget` with `meetingPosition()` pathfinding
- **Anti-stuck**: Tracks hit timeouts, resends move commands if no damage received

### 1.3 Key Discovery: rAthena Database is Complete

The `knowledge/rathena_db/` directory contains the **complete rAthena game database**:

| File | Size | Lines | Contents |
|---|---|---|---|
| `mob_db.yml` | 795KB | 42,537 | All monsters with stats, elements, drops, modes |
| `mob_skill_db.txt` | 481KB | 5,783 | All monster skills with conditions, rates, targets |
| `attr_fix.yml` | 8.5KB | 478 | Element damage multipliers for levels 1-4 |
| `skill_tree.yml` | 83KB | 3,579 | All skill trees with prerequisites |
| `size_fix.yml` | 1.4KB | 40 | Size damage modifiers per weapon type |
| `job_stats.yml` | 51KB | 2,930 | Job stat bonuses per level |
| `item_db_equip.yml` | Large | — | All equipment items |
| `item_db_etc.yml` | Large | — | All misc items |
| `item_db_usable.yml` | Large | — | All usable items |

**Critical finding — ElementLevel matters**: The element chart in `combat_tactics.py` uses **Level 1** values from `attr_fix.yml`. But most monsters have ElementLevel 2-4. The correct multipliers vary significantly by level:

| Attack → Defense | Lv1 | Lv2 | Lv3 | Lv4 |
|---|---|---|---|---|
| Water → Fire | 150% | 175% | 200% | 200% |
| Water → Wind | 50% | 25% | 0% | 0% |
| Holy → Undead | 150% | 175% | 200% | 200% |
| Holy → Dark | 125% | 150% | 175% | 200% |
| Ghost → Neutral | 25% | 25% | 0% | 0% |

**NOW**: The bot must read the monster's `ElementLevel` from `mob_db.yml` and use the correct level-specific chart. The `elemental_matrix.py` module must be updated to load all 4 levels from `attr_fix.yml`.

**Critical finding — NV_BASIC max level is 9**: The rAthena skill tree says `NV_BASIC` max level is **9**, not 1. Already fixed in `conscious_engine.py`.

**Critical finding — Monster database is complete**: The rAthena DB has all 2675 monsters with full stats, skills, drops, and modes. The hardcoded 87 monsters in `predictive_aggro.py` must be replaced by reading from `mob_db.yml` and `mob_skill_db.txt`.

**Critical finding — Skill trees have prerequisites**: `skill_tree.yml` has `Requires` fields — e.g., `SM_MAGNUM` requires `SM_BASH` level 5. The conscious engine must validate prerequisites before recommending skills.

```
┌──────────────────────────────────────────────────────────────────┐
│                        GAME CLIENTS (1-N)                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐      ┌──────────┐    │
│  │ OpenKore │  │ OpenKore │  │ OpenKore │  ... │ OpenKore │    │
│  │  Bot #1  │  │  Bot #2  │  │  Bot #3  │      │  Bot #N  │    │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘      └────┬─────┘    │
│       │              │              │                 │         │
│  ┌────┴─────┐  ┌────┴─────┐  ┌────┴─────┐      ┌────┴─────┐    │
│  │ Bridge   │  │ Bridge   │  │ Bridge   │      │ Bridge   │    │
│  │ (Perl)   │  │ (Perl)   │  │ (Perl)   │      │ (Perl)   │    │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘      └────┬─────┘    │
│       │              │              │                 │         │
│       └──────────────┴──────────────┴─────────────────┘         │
│                              │ HTTP (keep-alive, MsgPack)        │
└──────────────────────────────┼──────────────────────────────────┘
                               │
┌──────────────────────────────┼──────────────────────────────────┐
│                    SIDECAR (Python/FastAPI)                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  API Layer (FastAPI, 20+ routers)                        │   │
│  │  /v2/ingest, /v2/state, /v2/actions, /v2/fleet, ...     │   │
│  └──────────────────────────┬───────────────────────────────┘   │
│                             │                                    │
│  ┌──────────────────────────┼───────────────────────────────┐   │
│  │  MIDDLE LAYER (Tactical)                                  │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐    │   │
│  │  │ Combat   │ │ Economy  │ │ Fleet    │ │ Learning │    │   │
│  │  │ Loop     │ │ Engine   │ │ Coord.   │ │ System   │    │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘    │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐    │   │
│  │  │ Reflex   │ │ Pattern  │ │ Resource │ │ Death    │    │   │
│  │  │ Pipeline │ │ Detector │ │ Manager  │ │ Analysis │    │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘    │   │
│  └──────────────────────────┬───────────────────────────────┘   │
│                             │                                    │
│  ┌──────────────────────────┼───────────────────────────────┐   │
│  │  TOP-DOWN LAYER (Strategic)                                │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐    │   │
│  │  │ CrewAI   │ │ PDCA     │ │ Goal     │ │ Mission  │    │   │
│  │  │ Agents   │ │ Loop     │ │ Planner  │ │ Agent    │    │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘    │   │
│  │  ┌──────────────────────────────────────────────────┐  │   │
│  │  │  LLM Provider Router (DeepSeek, OpenAI, Ollama)   │  │   │
│  │  └──────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  PERSISTENCE                                               │   │
│  │  SQLite (openmemory.db) — episodic memory, learned data   │   │
│  │  knowledge/rathena_db/ — rAthena game data (2675 mobs)     │   │
│  └──────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

---

## 2. Core Architecture: Hybrid Bottom-Up + Top-Down

### The Design Principle

> **The bottom layer must function without the layers above it.**
> The top layers are advisors, not controllers.

### Decision Distribution

| Decision Type | Layer | Latency | Frequency | Approach |
|---|---|---|---|---|
| Dodge AoE | Bridge | <50ms | Per cast | Hardcoded reflex |
| Use potion | Bridge | <100ms | Per HP drop | Threshold reflex |
| Flee from danger | Bridge | <100ms | Per threat | Behavior tree |
| Cast buff | Bridge | <500ms | Per cooldown | Behavior tree |
| Interrupt caster | Bridge | <100ms | Per cast | Hardcoded reflex |
| Select skill | Sidecar | <200ms | Per target | Rotation system |
| Swap gear | Sidecar | <500ms | Per element | Elemental matrix |
| Restock supplies | Sidecar | <2s | Per 5min | Resource manager |
| Change map | Sidecar | <1s | Per 30s | Hunting zone manager |
| Party coordination | Sidecar | <1s | Per event | Fleet coordinator |
| Set farming goal | LLM | 1-3s | Per 5min | CrewAI agent |
| Analyze death | LLM | 2-5s | Per death | Death analysis |
| Plan build | LLM | 3-10s | Per level-up | Progression planner |
| MVP strategy | LLM | 2-5s | Per MVP | Tactical commander |

### Information Flow

```
OBSERVATIONS FLOW UP:
  Bridge ──(events)──▶ Sidecar ──(patterns)──▶ LLM

ADJUSTMENTS FLOW DOWN:
  LLM ──(goals)──▶ Sidecar ──(rule changes)──▶ Bridge

NEVER COMMANDS:
  No layer ever sends a direct command to a lower layer.
  The lower layer always has the final say.
```

### Degradation Modes

| Failure | Effect | Recovery |
|---|---|---|
| LLM API down | No strategy changes, no death analysis | Bot continues with current rules |
| Sidecar crash | No tactics, no learning | Bridge keeps running with last rules |
| Bridge crash | Bot goes blind | OpenKore auto-reconnects, bridge reloads |
| Network down | No HTTP to sidecar | Bridge runs in offline mode with cached rules |
| All down | Bot runs on OpenKore's built-in AI | Better than nothing |

---

## 3. Layer 1: Bridge (Bottom-Up Reflex Layer)

### File: `plugins/aiSidecarBridge/aiSidecarBridge.pl`

The bridge is a **Perl plugin** that runs inside each OpenKore instance. It is the bot's spinal cord — it handles everything that needs to happen in under 100ms.

### 3.1 Reflex System (19 Reflexes)

All reflexes are hardcoded in Perl. No LLM, no Python, no network calls. Each reflex has:

- **Condition** (when to fire)
- **Action** (what to do)
- **Cooldown** (how long to wait before firing again)
- **Priority** (which reflex wins if multiple want to fire)
- **Global cooldown category** (can't use two items at once)

#### Emergency Reflexes (Priority 100)

| # | Name | Condition | Action | Cooldown |
|---|---|---|---|---|
| 1 | Emergency Heal | HP < 50% | Use best heal item/skill | 200ms |
| 2 | Emergency Flee | HP < 15% + aggro | Flee to safe spot | 1s |
| 3 | Emergency Teleport | HP < 12% | Teleport away | 3s |
| 14 | Zonk | HP ≤ 0 or ≤ 5 | Sit immediately | 2s |
| 15 | Death Spike | Deaths % 5 == 0 | Notify sidecar | 2min |

#### Combat Reflexes (Priority 95)

| # | Name | Condition | Action | Cooldown |
|---|---|---|---|---|
| 9 | Interrupt Cast | Monster casting within 10 tiles | Bash/stun | 1.5s |
| 13 | High Aggro Surround | Aggro > 10 | Flee + teleport | 3s |
| 17 | Pre-Dodge | Monster casting dangerous AoE | Flee immediately | 2s |

#### Proactive Reflexes (Priority 80)

| # | Name | Condition | Action | Cooldown |
|---|---|---|---|---|
| 10 | Pre-Pot | Boss within 15 tiles | Pre-heal | 5s |
| 16 | Pre-Buff | Out of combat, HP > 80%, SP > 30% | Cast self-buffs | 15s |
| 18 | Auto-Sit Regen | Out of combat, HP < 60% | Sit | 5s |
| 19 | Potion Top-Off | Out of combat, HP 30-80% | Use heal item | 10s |

#### Awareness Reflexes (Priority 50)

| # | Name | Condition | Action | Cooldown |
|---|---|---|---|---|
| 4 | Aggro Warning | Aggro > 5 | Notify sidecar | 5s |
| 5 | Low SP | SP < 15% | Notify sidecar | 10s |
| 6 | GM Detection | GM/Admin within 15 tiles | Switch to manual | 1min |
| 7 | Weight Warning | Weight > 85% | Notify sidecar | 30s |
| 8 | Equipment Broken | Broken equipment | Notify sidecar | 1min |
| 11 | Bot Cooperation | HP < 50% + aggro + no heal | Request help | 5s |
| 12 | Party Low HP | Party member HP < 20% | Notify sidecar | 10s |

### 3.2 Behavior Tree (NOW)

The 19 `if` blocks must be replaced by a proper **behavior tree** with:

- **Selector nodes** (try children in order, first success wins)
- **Sequence nodes** (run all children in order, all must succeed)
- **Condition nodes** (check a condition)
- **Action nodes** (perform an action)
- **Decorator nodes** (cooldown, priority, inversion)

This makes the bridge's behavior **deterministic**, **testable**, and **provably correct**.

### 3.3 Snapshot System

The bridge sends a JSON snapshot to the sidecar every 500ms containing:

- **Vitals**: HP, SP, weight, level, job
- **Position**: map, x, y
- **Combat**: AI sequence, target, aggro count
- **Progression**: base/job level, exp, skill points, stat points
- **Skills**: all known skills with levels
- **Actors**: nearby monsters, players, NPCs (up to 24)
- **Raw**: char name, master, AI queue, death count, route stats

### 3.4 Anti-Detection

- Human-like reaction delay: 10-50ms (configurable)
- Random jitter on cooldowns: ±30%
- Random action delay: 50-200ms before non-critical actions
- No delay on emergency actions (survival first)

---

## 4. Layer 2: Sidecar (Middle Tactical Layer)

### File: `AI_sidecar/ai_sidecar/` (Python/FastAPI)

The sidecar is a **Python FastAPI server** that runs alongside the game clients. It is the bot's cerebellum — it handles tactics, pattern detection, and adaptation.

### 4.1 API Layer

20+ API routers organized by domain:

| Router | Purpose | Endpoints |
|---|---|---|
| `/v2/ingest` | Receive events from bridge | `POST /event`, `POST /chat`, `POST /config` |
| `/v2/state` | Bot state management | `GET /{bot_id}`, `POST /{bot_id}/update` |
| `/v2/actions` | Action queue | `POST /propose`, `GET /{bot_id}/next` |
| `/v2/fleet` | Fleet coordination | `POST /register`, `POST /order`, `GET /status` |
| `/v2/combat` | Combat decisions | `POST /tactics`, `POST /reflex` |
| `/v2/planner` | Strategic planning | `POST /plan`, `GET /{bot_id}/plan` |
| `/v2/reflex` | Reflex rule management | `POST /rule`, `GET /rules` |
| `/v2/health` | Health checks | `GET /`, `GET /ready`, `GET /live` |

### 4.2 Combat System

#### Combat Loop (`combat/combat_loop.py`)

Runs at 200ms intervals. Wires together:

1. **Threat Targeting** (`combat/threat_targeting.py`): Selects the best target based on element, distance, HP, aggro state
2. **Skill Rotation** (`combat/skill_rotation.py`): Executes the optimal skill sequence for the current target
3. **Elemental Matrix** (`combat/elemental_matrix.py`): Calculates damage multipliers for all element combinations
4. **Buff Maintenance** (`combat/buff_maintenance.py`): Tracks buff durations and recasts before expiry
5. **Gear Swapper** (`combat/gear_swapper.py`): Recommends gear sets based on target element and size
6. **Resource Manager** (`combat/resource_manager.py`): Tracks potion stock, weight, durability, farming duration
7. **Reflex Combat** (`combat/reflex_combat.py`): Hardcoded combat reflexes that bypass the LLM
8. **Action Executor** (`combat/action_executor.py`): Enqueues actions to the bridge

#### Combat Tactics (`combat_tactics.py`)

Per-class skill combos with correct pre-renewal RO mechanics:

- **Mage**: Frost Diver → Fire Bolt (freeze + 4x damage), NOT Cold Bolt (water = 25% vs water)
- **Wizard**: Storm Gust → Lord of Vermillion (AoE freeze + shock)
- **Archer**: Double Strafe while kiting (never stand still)
- **Hunter**: Double Strafe + Blitz Beat + trap positioning
- **Swordsman**: Bash spam (stun interrupts casts) + Magnum Break AoE
- **Knight**: Bowling Bash AoE + Spear Boomerang ranged opener
- **Thief**: Double Attack (passive) + Hiding for escape
- **Assassin**: Grimtooth (ranged) → Sonic Blow (finisher), NOT Sonic Blow → Grimtooth
- **Acolyte**: Heal vs undead (damage), Holy Light vs others
- **Priest**: Turn Undead (instant kill chance) + Heal sustain

#### Mechanical Intuition (`mechanical_intuition.py`)

Correct pre-renewal RO formulas:

- **Flee rate**: `95% - (monster_hit - player_flee)`, capped 5%-95%
- **ASPD**: `200 - weapon_delay + sqrt((agi²×0.02) + (dex²×0.02) + (agi+dex)×0.5) × (200-delay)/250`, capped 190
- **Cast reduction**: `DEX × 0.01` (1% per point), capped 50% from DEX alone, 70% total
- **Crit rate**: `LUK × 0.3 + 1`
- **Stat breakpoints**: Per-class for all 16 classes (Swordsman: STR 80/AGI 70/VIT 50; Mage: INT 99/DEX 30-50; Assassin: AGI 80/LUK 30; etc.)

#### Combat Instinct (`combat_instinct.py`)

50+ known dangerous monster skills with proper threat levels:

| Skill | Element | AoE? | Danger | Action |
|---|---|---|---|---|
| Storm Gust | Water | Yes | Critical | Dodge |
| Meteor Storm | Fire | Yes | Critical | Dodge |
| Hell's Judgement | Dark | Yes | Critical | Dodge |
| Earthquake | Earth | Yes | Critical | Dodge |
| Dark Breath | Dark | Yes | Critical | Dodge |
| Lord of Vermillion | Wind | Yes | Critical | Dodge |
| Fire Breath | Fire | Yes | High | Dodge |
| Thunder Breath | Wind | Yes | High | Dodge |
| Heaven's Drive | Holy | Yes | High | Dodge |
| Thunderstorm | Wind | Yes | High | Dodge |
| Fire Pillar | Fire | Yes | High | Dodge |
| Wide Stun | Neutral | Yes | High | Dodge |
| Wide Freeze | Water | Yes | High | Dodge |
| Fire Bolt | Fire | No | Medium | Pot |
| Cold Bolt | Water | No | Medium | Pot |
| Lightning Bolt | Wind | No | Medium | Pot |
| Poison Attack | Poison | No | Medium | Cure |
| Stun Attack | Neutral | No | High | Cure |
| Freeze Attack | Water | No | High | Cure |

#### Predictive Aggro (`predictive_aggro.py`)

**NOW**: Must be replaced with dynamic loading from `mob_db.yml` and `mob_skill_db.txt`. The rAthena database has all 2675 monsters with full stats. The hardcoded 87 monsters is a seed, not the source of truth.

Current hardcoded data includes:
- **Assist aggro**: Orc family (Warrior, Archer, Lady, Zombie, Skeleton), Goblin family, Thief Bug
- **Night aggro**: Zombie (passive during day)
- **Ranged monsters**: Orc Archer (chase range 18, was 15)
- **Boss monsters**: 30+ MVPs with aggro ranges up to 20 cells
- **59 maps** with spawn data and pre-calculated danger scores

**NOW**: The `mob_db.yml` has `Modes` fields including `Detector`, `Angry`, `Assist`, etc. The `mob_skill_db.txt` has all monster skills with conditions, rates, cast times, and targets. The bot must read these at startup and build the aggro database dynamically.

#### Risk Assessment (`risk_assessment.py`)

Multi-factor risk scoring:

- **HP/SP levels**: Low HP = +0.3, low SP = +0.2
- **Level gap**: +20 levels = +0.4, +10 levels = +0.2
- **MVP**: +0.3 risk, +0.4 reward
- **Escape available**: No escape = +0.2
- **Element disadvantage**: +0.15 (Fire vs Water, Earth vs Fire, etc.)
- **Aggro chain**: +0.2 (assist aggro active)
- **Map danger**: Up to +0.2 (scaled by pre-calculated danger)
- **Night time**: +0.1
- **Ranged monster**: +0.1
- **Monster with skills**: +0.1
- **Learned risk**: Adjusted from death outcomes

### 4.3 Economy System

#### Economic Engine (`economy/economic_engine.py`)

- **Item valuation**: Cards > slotted equipment > ores > healing items > junk
- **Market arbitrage**: Buy low, sell high across NPC shops and player vendors
- **Farming selector**: Choose maps based on drop value, not just exp
- **Vending automation**: Auto-vendor items, set prices based on market data
- **Supply chain**: Auto-restock potions, arrows, materials

#### Opportunity Cost (`economy/opportunity_cost.py`)

- **Zeny/hour calculation** for each map
- **Exp/hour vs zeny/hour** tradeoff analysis
- **Time-to-restock** cost estimation

### 4.4 Fleet Coordination

#### Fleet Coordinator (`fleet/fleet_coordinator.py`)

- **Bot registration**: Auto-register bots with role, class, level
- **Role assignment**: farmer, buffer, merchant, scout, crafter, woe_alt
- **Party formation**: Auto-form parties with optimal composition
- **Buff coordination**: Priest buffs all party members
- **Shared threat detection**: One bot sees danger, all bots react
- **Coordinated retreat/attack**: Synchronized movement
- **Resource sharing**: Zeny, items, potions across accounts

#### Multi-Account Synergy (`fleet/multi_account_synergy.py`)

- **Optimal party composition**: 1 tank + 1 healer + 2 DPS + 1 support
- **Level gap management**: Keep party within 10 levels for shared exp
- **Role rotation**: Switch roles based on who's online
- **AFK management**: Auto-replace AFK party members

#### Swarm AI (`fleet/swarm_ai.py`)

- **Decentralized coordination**: No single point of failure
- **Emergent behavior**: Simple rules produce complex group tactics
- **Auto-scaling**: Add/remove bots without reconfiguration

### 4.5 Learning System

#### Death Analysis (`learning/death_analysis.py`)

When a bot dies:

1. **Capture context**: Last 10 seconds of events, position, skills, aggro
2. **Identify cause**: Which monster, which skill, what was the bot doing
3. **Classify pattern**: AoE death, multi-hit death, status + follow-up, etc.
4. **Adjust behavior**: Update flee thresholds, add dodge rules, change map
5. **Share knowledge**: All bots learn from one bot's death

#### Shared Learning DB (`learning/shared_learning_db.py`)

- **SQLite-backed** persistent learning
- **Cross-bot knowledge sharing**: One bot learns, all benefit
- **Versioned knowledge**: Roll back bad learning
- **Confidence scoring**: Only apply high-confidence learnings

---

## 5. Layer 3: LLM / Conscious Engine (Top-Down Strategic Layer)

### 5.1 When the LLM is Called

The LLM is a **last resort**, not the default. It is only invoked for:

| Situation | Frequency | LLM Task |
|---|---|---|
| Bot just started | Once | "What's my build? Where should I go?" |
| Level up | Every 5-10 levels | "What skills should I learn next?" |
| Death | Per death | "Why did I die? What should I change?" |
| Stuck | Per 5min stuck | "I've been stuck for 5 minutes, what now?" |
| New monster | First encounter | "I've never seen this monster, what do I do?" |
| Map change | Per map | "Is this map safe? What should I hunt?" |
| Restock needed | Per 30min | "I need potions, where should I go?" |
| MVP spotted | Per MVP | "Can I take this MVP? What's the strategy?" |
| No progress | Per 10min | "I'm not gaining exp, what's wrong?" |

### 5.2 CrewAI Agent System

The LLM is accessed through a **CrewAI multi-agent system**:

| Agent | Role | Expertise |
|---|---|---|
| Strategic Planner | High-level goals | "Farm until level 70, then switch to Orcs" |
| Tactical Commander | Combat decisions | "Use Fire Bolt on Earth monsters" |
| Progression Planner | Build optimization | "Max INT first, then DEX" |
| Economy Agent | Resource management | "Sell these items, buy those" |
| Safety Agent | Risk assessment | "This map is too dangerous" |
| Navigation Agent | Pathfinding | "Go to Prontera via this route" |
| Social Agent | Player interaction | "Respond to whispers, join parties" |
| Fleet Liaison | Multi-bot coordination | "Priest, buff the Knight" |
| Resource Manager | Supply chain | "Restock potions at the shop" |
| Questing Agent | Quest automation | "Complete this quest chain" |
| Opportunistic Trader | Market manipulation | "Buy low, sell high" |

### 5.3 PDCA Loop

The **Plan-Do-Check-Act** loop runs at three frequencies:

| Loop | Interval | Purpose |
|---|---|---|
| Short-term | 5s | Combat adjustments, immediate threats |
| Medium-term | 30s | Map evaluation, resource check |
| Long-term | 120s | Goal reassessment, build planning |

### 5.4 Provider Router

Supports multiple LLM providers:

| Provider | Use Case | Cost |
|---|---|---|
| DeepSeek (default) | All strategic decisions | Low |
| OpenAI | Complex reasoning | Medium |
| Ollama (local) | Offline mode, privacy | Free |
| Custom | Any OpenAI-compatible API | Varies |

---

## 6. Transport: HTTP as Cross-Platform Backbone

### 6.1 Why HTTP

- **Cross-platform**: Works on Windows, Linux, macOS
- **Firewall-friendly**: Port 8080, no special permissions
- **SaaS-ready**: Same protocol works over the internet
- **Debugging**: curl, Postman, browser all work
- **Auth**: Standard HTTP auth (Bearer token)
- **Load balancing**: Standard HTTP load balancers

### 6.2 Performance Optimizations (NOW)

| Optimization | Current | Target | Impact |
|---|---|---|---|
| Connection | Open/close per request | Keep-alive (persistent) | 10x fewer TCP handshakes |
| Serialization | JSON | MessagePack or CBOR | 5-10x faster parsing |
| Polling | 500ms fixed | Adaptive (100ms-2s) | 5x faster in combat |
| Compression | None | gzip on large payloads | 10x smaller payloads |
| Batching | One event at a time | Batch events per tick | 10x fewer requests |
| Auth check | Every request | Cached token validation | Sub-ms auth |

### 6.3 Protocol

```
Bridge → Sidecar (POST /v2/ingest/event):
{
  "kind": "bridge_reflex",
  "reflex": "emergency_heal",
  "hp_ratio": 0.45,
  "hp": 450,
  "max_hp": 1000,
  "aggro_count": 3,
  "map": "gef_fild14",
  "timestamp": 1712345678000
}

Sidecar → Bridge (GET /v2/actions/{bot_id}/next):
{
  "action_id": "act_abc123",
  "command": "is White Potion",
  "priority": 100,
  "expires_at": "2026-07-15T12:00:05Z"
}
```

### 6.4 SaaS Architecture (FUTURE ONLY)

See Section 14. This is the only section marked FUTURE. Everything else must be implemented now.

---

## 7. Solo Gameplay Architecture

### 7.1 Leveling Flow

```
START → Town (buy supplies)
  ↓
Go to hunting map (sidecar recommends)
  ↓
Enter combat loop (bridge handles 95%)
  ↓
  ├─ Kill mobs → gain exp → level up
  │   ↓
  │  LLM: "Learn Fire Bolt level 5, add 5 INT"
  │   ↓
  │  Continue farming
  │
  ├─ HP low → potion (bridge reflex)
  │
  ├─ HP critical → flee/teleport (bridge reflex)
  │
  ├─ Out of potions → return to town (sidecar)
  │   ↓
  │  Restock → return to map
  │
  ├─ Weight full → return to town (sidecar)
  │   ↓
  │  Sell junk → restock → return
  │
  └─ Death → analyze (LLM)
      ↓
     Adjust behavior → continue
```

### 7.2 Combat Flow

```
IDLE → scan for targets (200ms tick)
  ↓
Target found → approach (pathfinding)
  ↓
In range → execute skill rotation
  ↓
  ├─ Skill fires → check result
  │   ├─ Hit → continue rotation
  │   ├─ Miss → try again or switch skill
  │   └─ Monster dead → next target
  │
  ├─ Monster casts → check skill
  │   ├─ Dangerous AoE → dodge (reflex, <50ms)
  │   ├─ Dangerous single-target → pot up (reflex, <100ms)
  │   └─ Harmless → ignore
  │
  ├─ HP drops → check threshold
  │   ├─ < 50% → potion (reflex)
  │   ├─ < 20% → flee (reflex)
  │   └─ < 10% → teleport (reflex)
  │
  └─ Multiple aggro → check count
      ├─ > 5 → AoE skills
      ├─ > 10 → flee + teleport
      └─ > 15 → emergency teleport
```

### 7.3 Class-Specific Behavior

Each class has a **behavior tree** that defines its role:

- **Mage**: Position at max range → Cast → If aggroed → Teleport → Reposition
- **Archer**: Keep distance → Shoot → If mob gets close → Arrow Shower (knockback) → Reposition
- **Knight**: Charge in → Bowling Bash → If surrounded → Magnum Break → If low HP → Retreat
- **Assassin**: Grimtooth from range → If mob closes → Sonic Blow → If surrounded → Venom Dust → Escape
- **Priest**: Stay behind party → Heal → Buff → If aggroed → Teleport → Return
- **Merchant**: Stay near town → Discount/Overcharge → Mammonite for damage → Pushcart for weight

---

## 8. Multi-Bot / Fleet Architecture

### 8.1 Fleet Composition

```
┌─────────────────────────────────────────────────────┐
│                    FLEET LEADER                       │
│  (Sidecar Fleet Coordinator)                          │
│  - Assigns roles                                      │
│  - Forms parties                                       │
│  - Coordinates movement                                │
│  - Shares intelligence                                 │
└───────────────────────────────────────────────────────┘
         │            │            │            │
         ▼            ▼            ▼            ▼
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│  Tank    │ │  Healer  │ │  DPS #1  │ │  DPS #2  │
│ (Knight) │ │ (Priest) │ │ (Wizard) │ │ (Hunter) │
└──────────┘ └──────────┘ └──────────┘ └──────────┘
```

### 8.2 Optimal Party Composition

| Role | Class | Responsibility |
|---|---|---|
| Tank | Knight/Paladin | Hold aggro, absorb damage, Bowling Bash AoE |
| Healer | Priest/Arch Bishop | Heal, buff (Blessing, Increase AGI), resurrect |
| AoE DPS | Wizard/High Wizard | Storm Gust, Meteor Storm, Lord of Vermillion |
| Single DPS | Hunter/Sniper | Double Strafe, Blitz Beat, trap support |
| Support | Bard/Dancer | Songs, dances, SP regen, stat boosts |
| Flex | Assassin/Rogue | Steal, backstab, scout, emergency DPS |

### 8.3 Fleet Coordination Protocol

```
1. FLEET LEADER broadcasts: "Forming party at prontera 150,150"
2. All bots move to position
3. FLEET LEADER: "Priest, buff the party"
4. Priest casts Blessing, Increase AGI on all members
5. FLEET LEADER: "Move to gef_fild14"
6. All bots path to map
7. FLEET LEADER: "Tank, pull the Orc Warriors"
8. Tank charges in, uses Bowling Bash
9. Wizard casts Storm Gust on grouped mobs
10. Hunter picks off stragglers
11. Priest heals tank, rebuffs as needed
12. If any bot dies → FLEET LEADER: "Retreat to safe spot"
13. Priest resurrects dead bot
14. Resume farming
```

### 8.4 Shared Intelligence

| Intelligence | Source | Distribution | Update |
|---|---|---|---|
| Map danger | Any bot that visits | All bots | Real-time |
| MVP spawn | Any bot that sees it | All bots | Real-time |
| PK warning | Any bot that sees PKer | All bots | Real-time |
| Market prices | Merchant bot | All bots | Per hour |
| Death patterns | Any bot that dies | All bots | Per death |
| Safe spots | Any bot that finds one | All bots | Per discovery |

---

## 9. Combat System Deep Dive

### 9.1 Skill Execution Pipeline

```
1. SELECT TARGET (threat_targeting.py)
   - Priority: current target > nearest aggro > nearest mob
   - Element check: prefer mobs we have element advantage against
   - Level check: skip mobs > 20 levels above or below

2. SELECT SKILL (skill_rotation.py)
   - Check current rotation
   - Check element advantage
   - Check SP availability
   - Check cooldowns
   - Check range
   - Return best skill

3. EXECUTE SKILL (action_executor.py)
   - Enqueue to bridge
   - Bridge sends command to OpenKore
   - OpenKore sends packet to server
   - Server processes skill

4. CHECK RESULT (combat_loop.py)
   - Did the skill fire? (check cast bar)
   - Did it hit? (check damage numbers)
   - Is the target dead? (check HP)
   - Is the target still casting? (check cast bar)

5. ADAPT (reflex_combat.py)
   - If skill missed → try again or switch
   - If target dead → next target
   - If monster casting → check if we need to dodge
   - If HP low → potion
```

### 9.2 Elemental Advantage System (NOW: Must use ElementLevel)

The current chart uses Level 1 values. The bot must read the monster's `ElementLevel` from `mob_db.yml` and use the correct level-specific chart from `attr_fix.yml`.

**Level 1 values (current, used for ElementLevel 1 monsters):**

| Attack → Defense | Neutral | Water | Earth | Fire | Wind | Poison | Holy | Dark | Ghost | Undead |
|---|---|---|---|---|---|---|---|---|---|---|
| Neutral | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 25% | 100% |
| Water | 100% | 25% | 100% | 150% | 50% | 100% | 75% | 100% | 100% | 100% |
| Earth | 100% | 100% | 100% | 50% | 150% | 100% | 75% | 100% | 100% | 100% |
| Fire | 100% | 50% | 150% | 25% | 100% | 100% | 75% | 100% | 100% | 125% |
| Wind | 100% | 175% | 50% | 100% | 25% | 100% | 75% | 100% | 100% | 100% |
| Poison | 100% | 100% | 125% | 125% | 125% | 0% | 75% | 50% | 100% | -25% |
| Holy | 100% | 100% | 100% | 100% | 100% | 100% | 0% | 125% | 100% | 150% |
| Dark | 100% | 100% | 100% | 100% | 100% | 50% | 125% | 0% | 100% | -25% |
| Ghost | 25% | 100% | 100% | 100% | 100% | 100% | 75% | 75% | 125% | 100% |
| Undead | 100% | 100% | 100% | 100% | 100% | 50% | 100% | 0% | 100% | 0% |

**Level 4 values (used for ElementLevel 4 monsters like Osiris):**

| Attack → Defense | Neutral | Water | Earth | Fire | Wind | Poison | Holy | Dark | Ghost | Undead |
|---|---|---|---|---|---|---|---|---|---|---|
| Water | 100% | -50% | 100% | 200% | 0% | 75% | 0% | 25% | 100% | 150% |
| Fire | 100% | 0% | 200% | -50% | 100% | 75% | 0% | 25% | 100% | 200% |
| Holy | 100% | 75% | 75% | 75% | 75% | 125% | -100% | 200% | 100% | 200% |
| Ghost | 0% | 25% | 25% | 25% | 25% | 25% | 0% | 0% | 200% | 175% |

**NOW**: The `elemental_matrix.py` module must load all 4 levels from `attr_fix.yml` and select the correct level based on the monster's `ElementLevel` field.

### 9.3 MVP Mechanics

Each MVP has a **mechanics profile**:

| MVP | Element | Level | HP | Dangerous Skills | Strategy |
|---|---|---|---|---|---|
| Osiris | Undead Lv4 | 78 | 415,400 | Dark Breath (AoE), Hell's Judgement | Stay at range, use Holy |
| Baphomet | Demon Lv3 | 81 | — | Hell's Judgement (massive AoE) | Run when casting, use Holy |
| Maya | Earth Lv3 | 65 | — | Reflect Shield (physical reflect) | Use magic, NOT melee |
| Eddga | Brute Lv1 | 65 | — | Earthquake (AoE stun) | Need VIT 80+, use Wind |
| Doppelganger | Demon Lv2 | 77 | — | Bowling Bash (AoE) | Don't group up, use Holy |
| Orc Lord | Demi-Human Lv2 | 75 | — | Bowling Bash, Bash | Tank with high VIT, use Fire |
| Mistress | Demon Lv3 | 63 | — | Thunderstorm, Storm Gust | Use Fire, interrupt casts |
| Moonlight Flower | Demon Lv2 | 62 | — | Soul Strike, Napalm Beat | Use Holy, interrupt |
| Dracula | Demon Lv3 | 68 | — | Hell's Judgement, Dark Breath | Stay at range, use Holy |
| Bloody Knight | Demi-Human Lv2 | 70 | — | Bowling Bash, Bash | Tank, use Fire |

**NOW**: MVP data must be loaded from `mob_db.yml` and `mob_skill_db.txt` dynamically, not hardcoded. The `mob_db.yml` has `Class: Boss` for all MVPs with full stats.

### 9.4 Potion Management

```
Potion Cooldown: 2 seconds (pre-renewal)

Usage Rules:
1. HP < 50%: Use best available heal item (bridge reflex)
2. HP < 80% and out of combat: Top off (bridge reflex, 10s cooldown)
3. Before boss fight: Pre-pot (bridge reflex, 5s cooldown)
4. Never spam: Track cooldown, wait 2s between potions
5. Stock management: Restock when < 20 potions remaining

Potion Priority:
1. Config-pushed items (class-aware, dynamic)
2. Config-pushed skills (Heal, etc.)
3. Hardcoded fallback: White Potion (always available)
```

---

## 10. Economic System

### 10.1 Item Valuation

| Category | Value | Action |
|---|---|---|
| Cards | High | Keep |
| Slotted equipment | High | Keep |
| Refining materials (Elunium, Oridecon) | High | Keep |
| Quest items | High | Keep |
| Healing items | Medium | Keep |
| Unsorted equipment | Medium | Sell |
| Junk | Low | Sell to NPC |

### 10.2 Farming Economics

```
Zeny/Hour = (Average drop value × kills per hour) - (Potion cost per hour)

The bot chooses the map with the best zeny/hour, not just the best exp/hour.
```

**NOW**: The `mob_db.yml` has complete drop data for all 2675 monsters with rates. The bot must use this to calculate expected zeny/hour per map.

### 10.3 Restock Logic

```
Restock when:
- Potion stock < 20
- Arrow stock < 200
- Weight > 85%
- Equipment durability < 30%

Restock plan:
1. Return to town
2. Sell junk to NPC
3. Buy potions (up to 200)
4. Buy arrows (up to 1000)
5. Repair equipment
6. Return to farming map
```

---

## 11. Learning & Adaptation

### 11.1 What the Bot Learns

| Knowledge | Source | Persistence | Sharing |
|---|---|---|---|
| Monster aggro behavior | Observation | SQLite | All bots |
| Map danger scores | Death analysis | SQLite | All bots |
| Optimal skill rotations | Trial and error | SQLite | Per class |
| Potion consumption rate | Tracking | SQLite | Per bot |
| Market prices | Vending data | SQLite | All bots |
| MVP spawn timers | Observation | SQLite | All bots |
| Safe spots | Discovery | SQLite | All bots |
| PKer names | Observation | SQLite | All bots |

### 11.2 Death Analysis Pipeline

```
1. CAPTURE: Last 10 seconds of events
2. CLASSIFY: What killed the bot?
   - AoE skill (Storm Gust, Meteor Storm, etc.)
   - Multi-hit (rapid consecutive attacks)
   - Status + follow-up (stun → kill, freeze → kill)
   - Level gap (mob too strong)
   - Element disadvantage
   - Aggro overwhelm (too many mobs)
3. ADJUST:
   - Add dodge rule for the specific skill
   - Increase flee threshold
   - Change map
   - Level up before returning
4. SHARE: Broadcast to all bots
```

### 11.3 Confidence System

```
Confidence = (successful observations) / (total observations)

Only apply learned rules with confidence > 0.7
Roll back rules that cause more deaths
```

---

## 12. Observability & Safety

### 12.1 Metrics

| Metric | Source | Purpose |
|---|---|---|
| HP/SP over time | Bridge | Health monitoring |
| Deaths per hour | Bridge | Safety alert |
| Exp per hour | Sidecar | Efficiency tracking |
| Zeny per hour | Sidecar | Economic tracking |
| Potions used per hour | Sidecar | Resource tracking |
| Reflex fires per minute | Bridge | Combat intensity |
| LLM calls per hour | Sidecar | Cost tracking |
| Fleet sync latency | Sidecar | Coordination health |

### 12.2 Safety Systems

| System | Trigger | Action |
|---|---|---|
| Circuit breaker | 5 deaths in 10 minutes | Stop all bots, notify user |
| GM detection | GM/Admin within 15 tiles | Switch to manual mode |
| Anti-detection | Random delays, jitter | Avoid bot detection |
| Degradation | Component failure | Graceful fallback |
| Rate limiting | Too many actions | Slow down, avoid server flags |

### 12.3 Audit Logging

Every significant action is logged with:
- **Timestamp**: When it happened
- **Bot ID**: Which bot
- **Action**: What was done
- **Context**: Why it was done
- **Outcome**: What happened as a result

---

## 13. Windows Compatibility

### 13.1 What Works on Windows

| Component | Windows Status | Notes |
|---|---|---|
| OpenKore | ✅ Full support | Originally a Windows tool |
| Perl bridge | ✅ Full support | Strawberry Perl or ActivePerl |
| Python sidecar | ✅ Full support | Standard Python |
| HTTP transport | ✅ Full support | Cross-platform by design |
| SQLite | ✅ Full support | Python stdlib |
| All 19 reflexes | ✅ Full support | Pure Perl, no platform deps |
| All combat modules | ✅ Full support | Pure Python |
| Fleet coordination | ✅ Full support | HTTP-based |

### 13.2 What's Different on Windows

| Feature | Linux | Windows |
|---|---|---|
| Shared memory | ✅ mmap | ❌ Not available |
| Unix sockets | ✅ Fast | ❌ Limited |
| Process signaling | ✅ SIGTERM, SIGKILL | ⚠️ Taskkill |
| File paths | `/path/to/file` | `C:\path\to\file` |
| Line endings | LF | CRLF |
| Case sensitivity | Case-sensitive | Case-insensitive |

### 13.3 Windows Optimization Path

For Windows users who want maximum performance:

1. **Keep HTTP** (works great on Windows)
2. **Use keep-alive** (reduces TCP overhead)
3. **Use MessagePack** (faster than JSON, pure Python library)
4. **Reduce polling interval** to 100ms in combat (still fast enough)
5. **Batch events** (send 5-10 events per request instead of 1)

---

## 14. SaaS Architecture (FUTURE ONLY)

> **This is the ONLY section marked FUTURE. Everything else in this document must be implemented NOW.**

### 14.1 Multi-Tenant Design

```
┌─────────────────────────────────────────────────────────┐
│                    CLOUD SIDECAR                          │
│                                                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │ User A       │  │ User B       │  │ User C       │     │
│  │ 3 bots       │  │ 5 bots       │  │ 1 bot        │     │
│  │ $15/month    │  │ $25/month    │  │ $5/month     │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
│                                                           │
│  ┌─────────────────────────────────────────────────────┐ │
│  │  Shared LLM Pool (cost amortized across users)      │ │
│  │  Shared Knowledge DB (anonymized)                   │ │
│  │  Shared Market Data (all users contribute)           │ │
│  └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

### 14.2 SaaS Benefits

- **Zero setup**: Users don't install anything
- **Auto-scaling**: More bots = more resources
- **Shared intelligence**: All users benefit from collective learning
- **Centralized updates**: Push new features instantly
- **Billing flexibility**: Per-bot, per-hour, or flat rate
- **Monitoring**: Dashboards, alerts, support

### 14.3 SaaS Challenges

- **Latency**: Internet adds 10-50ms vs localhost
- **Reliability**: Need 99.9% uptime
- **Security**: User authentication, data isolation
- **Cost**: LLM API costs scale with users
- **Compliance**: Game ToS, data privacy

---

## 15. Issue Registry: Every Problem Addressed

### 15.1 Architecture Issues

| # | Issue | Fix | Status |
|---|---|---|---|
| 1 | 500ms snapshot loop too slow for combat | Adaptive polling (100ms in combat, 2s idle) | NOW |
| 2 | Bridge waits for sidecar before acting | Bridge acts independently, reports upward | ✅ Done |
| 3 | Python sidecar adds HTTP latency | Keep-alive + MessagePack + batching | NOW |
| 4 | No priority system for reflexes | Behavior tree with priority nodes | NOW |
| 5 | State scattered across components | SQLite for persistence, bridge for real-time | NOW |
| 6 | No crash safety | Each layer functions without layers above | ✅ Done |
| 7 | No Windows support for shared memory | HTTP is cross-platform, shared memory is optional | NOW |

### 15.2 Combat Issues

| # | Issue | Fix | Status |
|---|---|---|---|
| 8 | Mage uses Cold Bolt vs water (25% damage) | Frost Diver → Fire Bolt (freeze + 4x) | ✅ Done |
| 9 | Archer stands still while shooting | Always kite (should_kite returns True for ranged) | ✅ Done |
| 10 | Assassin uses Sonic Blow → Grimtooth (wrong order) | Grimtooth (ranged) → Sonic Blow (finisher) | ✅ Done |
| 11 | Flee formula calculates value, not rate | 95% - (hit - flee), capped 5-95% | ✅ Done |
| 12 | ASPD formula off by 2-3x | Weapon delay + sqrt stat mod, capped 190 | ✅ Done |
| 13 | Cast reduction formula wrong (DEX*0.02) | DEX*0.01, cap 50% from DEX, 70% total | ✅ Done |
| 14 | Crit rate formula wrong | LUK*0.3 + 1 | ✅ Done |
| 15 | No monster skill awareness | 50+ dangerous skills with threat levels | ✅ Done |
| 16 | No multi-hit detection | 3+ damage events in <1s = flee | ✅ Done |
| 17 | No element disadvantage tracking | Element disadvantage risk factor | ✅ Done |
| 18 | No assist aggro awareness | Orc/Goblin/Thief Bug families | ✅ Done |
| 19 | No night aggro awareness | Zombie marked as night-only | ✅ Done |
| 20 | No ranged monster awareness | Orc Archer chase=18, ranged flag | ✅ Done |
| 21 | No potion cooldown tracking | 2s potion cooldown enforced | NOW |
| 22 | No skill delay tracking | Cast time + delay + cooldown per skill | NOW |
| 23 | No gear swapping | Gear swapper module with elemental sets | NOW |
| 24 | No MVP mechanics | 30+ MVP profiles with strategies | NOW |
| 25 | No spawn control | Spawn timer tracking, position optimization | NOW |
| 26 | No map geometry awareness | Wall/obstacle detection, line-of-sight breaks | NOW |
| 27 | Element chart uses Level 1 only | Load all 4 levels from attr_fix.yml, use monster's ElementLevel | NOW |
| 28 | Monster database 96% incomplete (87 of 2675) | Load from mob_db.yml and mob_skill_db.txt dynamically | NOW |
| 29 | Skill trees don't validate prerequisites | Load from skill_tree.yml, check Requires before recommending | NOW |

### 15.3 Strategy Issues

| # | Issue | Fix | Status |
|---|---|---|---|
| 30 | LLM called for every decision | LLM is last resort, not default | NOW |
| 31 | No class-specific behavior trees | Per-class behavior trees (Mage, Archer, etc.) | NOW |
| 32 | No party synergy | Fleet coordinator with role assignment | NOW |
| 33 | No economic awareness | Market prices, zeny/hour optimization | NOW |
| 34 | No level penalty awareness | Level penalty in hunting recommendations | NOW |
| 35 | No learning from death | Death analysis pipeline | NOW |
| 36 | No cross-bot learning | Shared learning DB | NOW |
| 37 | No build planning | Conscious engine with per-class builds | ✅ Done |
| 38 | No skill learn order optimization | Correct order for all classes | ✅ Done |
| 39 | No stat distribution optimization | Per-class stat priorities | ✅ Done |
| 40 | NV_BASIC max level was 1 (should be 9) | Fixed to 9 per rAthena skill_tree.yml | ✅ Done |

### 15.4 Multi-Bot Issues

| # | Issue | Fix | Status |
|---|---|---|---|
| 41 | No party formation | Auto-form parties with optimal composition | NOW |
| 42 | No buff coordination | Priest buffs all party members | NOW |
| 43 | No shared threat detection | One bot sees danger, all react | NOW |
| 44 | No coordinated retreat | Fleet leader commands retreat | NOW |
| 45 | No resource sharing | Zeny/items shared across accounts | NOW |
| 46 | No role assignment | Farmer, buffer, merchant, scout, etc. | NOW |
| 47 | No level gap management | Keep party within 10 levels | NOW |

### 15.5 Technical Debt Issues

| # | Issue | Fix | Status |
|---|---|---|---|
| 48 | No test coverage for bridge | Perl test harness | NOW |
| 49 | No integration tests | Bridge + sidecar together | NOW |
| 50 | No performance benchmarks | Reflex latency targets (<1ms) | NOW |
| 51 | No structured logging | JSON logging | NOW |
| 52 | No metrics | Prometheus metrics | NOW |
| 53 | No tracing | OpenTelemetry | NOW |
| 54 | No alerting | Death spike, stuck, no exp | NOW |
| 55 | No config validation | Schema checking | NOW |
| 56 | No single config source | YAML with inheritance | NOW |
| 57 | Skills referenced as strings | Skill registry with objects | NOW |
| 58 | HTTP open/close per request | Keep-alive connections | NOW |
| 59 | JSON serialization overhead | MessagePack for high-frequency data | NOW |
| 60 | SQLite files tracked in git | Added to .gitignore | ✅ Done |
| 61 | .pids/ not in gitignore | Added to .gitignore | ✅ Done |
| 62 | sidecar_auth_token.txt not in gitignore | Added to .gitignore | ✅ Done |

---

## Appendix A: File Map

```
openkore-ai-v3/
├── plugins/
│   └── aiSidecarBridge/
│       └── aiSidecarBridge.pl          # Bridge (Perl, 19 reflexes, 3425 lines)
│
├── AI_sidecar/
│   └── ai_sidecar/
│       ├── app.py                      # FastAPI entry point
│       ├── config.py                   # Configuration
│       ├── combat_tactics.py           # Per-class skill combos
│       ├── mechanical_intuition.py     # RO formulas (flee, ASPD, cast, crit)
│       ├── combat_instinct.py          # Monster skill awareness
│       ├── predictive_aggro.py         # Monster aggro database
│       ├── risk_assessment.py          # Risk/reward scoring
│       ├── conscious_engine.py         # Build plans, skill learn order
│       ├── game_engine.py              # rAthena knowledge integration
│       ├── combat/
│       │   ├── combat_loop.py          # Main combat loop (200ms)
│       │   ├── reflex_combat.py        # Hardcoded combat reflexes
│       │   ├── skill_rotation.py       # Skill selection and rotation
│       │   ├── elemental_matrix.py     # Element advantage calculations
│       │   ├── buff_maintenance.py     # Buff tracking and recasting
│       │   ├── gear_swapper.py         # Dynamic gear changes
│       │   ├── resource_manager.py     # Potion/consumable management
│       │   ├── threat_targeting.py     # Target selection
│       │   ├── action_executor.py      # Action enqueueing
│       │   ├── mvp_mechanics.py        # MVP skill/phase knowledge
│       │   ├── mvp_tracker.py          # MVP spawn tracking
│       │   ├── anti_killsteal.py       # Killsteal prevention
│       │   ├── safe_position.py        # Safe spot management
│       │   ├── gather_and_kill.py      # Group combat tactics
│       │   ├── humanizer.py            # Human-like behavior
│       │   ├── gm_detector.py          # GM detection
│       │   ├── predictive_threat.py    # Threat prediction
│       │   ├── skill_chain_executor.py # Skill chain execution
│       │   ├── build_manager.py        # Build-aware decisions
│       │   └── woe_*.py               # WoE-specific tactics
│       ├── fleet/
│       │   ├── fleet_coordinator.py    # Multi-bot coordination
│       │   ├── multi_account_synergy.py # Party composition
│       │   ├── swarm_ai.py             # Decentralized coordination
│       │   ├── party_coordinator.py    # Party management
│       │   ├── role_manager.py         # Role assignment
│       │   ├── conflict_resolver.py    # Order conflict resolution
│       │   ├── cross_bot_resource_manager.py # Shared resources
│       │   └── self_learning.py        # Cross-bot learning
│       ├── economy/
│       │   ├── economic_engine.py      # Core economy
│       │   ├── farming_selector.py     # Map selection by value
│       │   ├── market_arbitrage.py     # Buy low, sell high
│       │   ├── vending_automation.py   # Auto-vending
│       │   ├── supply_chain.py         # Restock planning
│       │   └── opportunity_cost.py     # Zeny/hour optimization
│       ├── learning/
│       │   ├── death_analysis.py       # Post-mortem analysis
│       │   ├── shared_learning_db.py   # Cross-bot knowledge
│       │   └── strategy_optimizer.py  # Strategy refinement
│       ├── reflex/
│       │   ├── reflex_pipeline.py      # Reflex action pipeline
│       │   ├── rule_engine.py          # Reflex rule management
│       │   ├── trigger_matcher.py     # Condition matching
│       │   ├── circuit_breaker.py      # Safety circuit breaker
│       │   └── healing_optimizer.py   # Heal efficiency
│       ├── autonomy/
│       │   ├── pdca_loop.py            # Plan-Do-Check-Act loop
│       │   ├── goal_planner.py         # Goal decomposition
│       │   ├── mission_agent.py        # Mission execution
│       │   └── progress_tracker.py     # Progress monitoring
│       ├── crewai/
│       │   ├── crew_manager.py          # CrewAI orchestration
│       │   └── agents/*.py            # 15+ specialized agents
│       └── api/routers/*.py            # 20+ API routers
│
├── knowledge/
│   └── rathena_db/                     # rAthena game data
│       ├── db/pre-re/                  # Pre-renewal data
│       │   ├── mob_db.yml              # 2675 monsters (795KB, 42,537 lines)
│       │   ├── mob_skill_db.txt        # Monster skills (481KB, 5,783 lines)
│       │   ├── skill_tree.yml          # Skill trees (83KB, 3,579 lines)
│       │   ├── item_db_equip.yml       # Equipment items
│       │   ├── item_db_etc.yml         # Misc items
│       │   ├── item_db_usable.yml      # Usable items
│       │   ├── job_stats.yml           # Job stat bonuses (51KB, 2,930 lines)
│       │   ├── size_fix.yml            # Size damage modifiers (1.4KB, 40 lines)
│       │   ├── attr_fix.yml            # Element damage modifiers (8.5KB, 478 lines)
│       │   └── level_penalty.yml       # Exp penalty by level
│       └── db/re/                      # Renewal data (same structure)
│
└── control/
    ├── ai_sidecar.txt                  # Bridge config
    ├── ai_sidecar_policy.txt           # Bridge policy
    └── sidecar_auth_token.txt          # Auto-generated auth token
```

---

## Appendix B: Key Metrics

| Metric | Current | Target | Status |
|---|---|---|---|
| Reflex latency | ~50ms (bridge) | <50ms | ✅ Done |
| Combat tick | 200ms | 100ms | NOW |
| Snapshot interval | 500ms | Adaptive (100ms-2s) | NOW |
| LLM call frequency | Per decision | Per 5-30min | NOW |
| Monsters known | 87 | 2675 (from mob_db.yml) | NOW |
| Maps known | 59 | All (from mob_db.yml) | NOW |
| Reflex count | 19 | Behavior tree (unlimited) | NOW |
| Party size | 1 | 1-12 | NOW |
| Fleet size | 3 | Unlimited | NOW |
| Death recovery | Manual | Automatic | NOW |
| Learning | None | Continuous | NOW |
| Windows support | Partial | Full | NOW |
| Element chart | Level 1 only | All 4 levels from attr_fix.yml | NOW |
| Skill validation | None | Prerequisites from skill_tree.yml | NOW |
| SaaS | — | FUTURE ONLY | 🔄 Future |

---

*This document represents the architecture as of 2026-07-15. Items marked "✅ Done" are implemented and committed. Items marked "NOW" must be implemented immediately. Only SaaS (Section 14) is marked "FUTURE".*
