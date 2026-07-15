# openkore-ai-v3 Architecture Document

> **Version:** 3.0  
> **Date:** 2026-07-15  
> **Author:** Pro RO Player / AI Engineer  
> **Scope:** Full-stack architecture for solo and multi-bot RO automation  
> **Honest Status:** 19 items truly done. 42 items marked "NOW" that haven't been started. 0 code changes since this doc was written. This is a confession, not a plan.

---

## Table of Contents

1. [The Truth: What's Actually Done vs What's Promised](#1-the-truth-whats-actually-done-vs-whats-promised)
2. [Core Architecture: Hybrid Bottom-Up + Top-Down](#2-core-architecture-hybrid-bottom-up--top-down)
3. [Layer 1: Bridge (Still 234 If Blocks, Not a Behavior Tree)](#3-layer-1-bridge-still-234-if-blocks-not-a-behavior-tree)
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
15. [Issue Registry: Honest Status of Every Problem](#15-issue-registry-honest-status-of-every-problem)
16. [The 8 Things That Can Be Fixed in One Day](#16-the-8-things-that-can-be-fixed-in-one-day)

---

## 1. The Truth: What's Actually Done vs What's Promised

This document was written as a plan. Then no code was written to execute the plan. This section is the **honest audit** of what the codebase actually contains vs what the doc claims.

### Actually Implemented (19 items, verified by code audit)

| # | What | File | Lines |
|---|---|---|---|
| 1 | Mage vs Water: Frost Diver → Fire Bolt | `combat_tactics.py` | 30-40 |
| 2 | Archer always kites | `combat_tactics.py` | 170-175 |
| 3 | Assassin: Grimtooth → Sonic Blow | `combat_tactics.py` | 100-110 |
| 4 | Flee formula: 95% - (hit - flee), capped 5-95% | `mechanical_intuition.py` | 148-160 |
| 5 | ASPD formula: weapon delay + sqrt stat mod | `mechanical_intuition.py` | 155-170 |
| 6 | Cast reduction: DEX*0.01, cap 50% | `mechanical_intuition.py` | 160-175 |
| 7 | Crit rate: LUK*0.3 + 1 | `mechanical_intuition.py` | 180-185 |
| 8 | 50+ dangerous monster skills | `combat_instinct.py` | 30-80 |
| 9 | Multi-hit detection | `combat_instinct.py` | 85-95 |
| 10 | Element disadvantage risk factor | `risk_assessment.py` | 60-70 |
| 11 | Assist aggro (Orc/Goblin/Thief Bug) | `predictive_aggro.py` | 60-70 |
| 12 | Night aggro (Zombie) | `predictive_aggro.py` | 90 |
| 13 | Ranged monster (Orc Archer chase=18) | `predictive_aggro.py` | 70 |
| 14 | 4 proactive bridge reflexes (pre-buff, pre-dodge, auto-sit, top-off) | `aiSidecarBridge.pl` | 3400-3550 |
| 15 | Phase-based build system (6 classes, 2-3 variants each) | `conscious_engine.py` | 60-1200 |
| 16 | Efficiency breakpoints for 39 skills | `conscious_engine.py` | 60-200 |
| 17 | Game mode → build variant selection | `conscious_engine.py` | 1250-1280 |
| 18 | .gitignore (openmemory.db, .pids/, auth token) | `.gitignore` | All |
| 19 | Architecture doc v3.0 (this document) | `ARCHITECTURE.md` | All |

### Promised But Not Started (42 items, all marked "NOW")

| Category | Count | Examples |
|---|---|---|
| Bridge architecture | 3 | Behavior tree, priority arbitration, global cooldowns |
| HTTP transport | 4 | Keep-alive, MessagePack, adaptive polling, batching |
| Combat features | 8 | Element chart all 4 levels, monster DB from mob_db.yml, potion CD 2s, skill delay enforcement, gear swap, spawn control, map geometry, skill tree validation |
| Fleet features | 7 | Party formation, buff coordination, threat sharing, retreat, resource sharing, role assignment, level gap management |
| Learning | 3 | Death analysis wiring, cross-bot learning, strategy optimization |
| Infrastructure | 12 | Tests, benchmarks, logging, metrics, tracing, alerting, config validation, single config source, skill registry |
| Strategy | 5 | LLM as last resort, class behavior trees, economy, level penalty, party synergy |

### The Gap

```
✅ Done:   19 items  ████████████░░░░░░░░░░░░░░░░  31%
📋 Promised: 42 items  ████████████████████████████  69%
🛠️  Started since doc: 0 items  ░░░░░░░░░░░░░░░░░░░░   0%
```

The doc was written. Zero code changes have been made to execute it. This is a **wish list**, not a **deliverable**.

---

## 2. Core Architecture: Hybrid Bottom-Up + Top-Down

### The Design Principle

> **The bottom layer must function without the layers above it.**
> The top layers are advisors, not controllers.

### Decision Distribution

| Decision Type | Layer | Latency | Frequency | Approach | Status |
|---|---|---|---|---|---|
| Dodge AoE | Bridge | <50ms | Per cast | Hardcoded reflex | ✅ Done |
| Use potion | Bridge | <100ms | Per HP drop | Threshold reflex | ⚠️ Cooldown wrong (500ms, should be 2s) |
| Flee from danger | Bridge | <100ms | Per threat | 234 if blocks | ❌ Not a behavior tree |
| Cast buff | Bridge | <500ms | Per cooldown | 234 if blocks | ❌ Not a behavior tree |
| Interrupt caster | Bridge | <100ms | Per cast | Hardcoded reflex | ✅ Done |
| Select skill | Sidecar | <200ms | Per target | Rotation system | ⚠️ Cast times defined but not enforced |
| Swap gear | Sidecar | <500ms | Per element | Elemental matrix | ❌ Not wired to combat loop |
| Restock supplies | Sidecar | <2s | Per 5min | Resource manager | ❌ Not tested |
| Change map | Sidecar | <1s | Per 30s | Hunting zone manager | ❌ Not tested |
| Party coordination | Sidecar | <1s | Per event | Fleet coordinator | ❌ Not tested |
| Set farming goal | LLM | 1-3s | Per 5min | CrewAI agent | ❌ Not tested |
| Analyze death | LLM | 2-5s | Per death | Death analysis | ❌ Module exists, not wired |
| Plan build | LLM | 3-10s | Per level-up | Progression planner | ❌ Not tested |
| MVP strategy | LLM | 2-5s | Per MVP | Tactical commander | ❌ Not tested |

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

| Failure | Effect | Recovery | Status |
|---|---|---|---|
| LLM API down | No strategy changes, no death analysis | Bot continues with current rules | ✅ Designed |
| Sidecar crash | No tactics, no learning | Bridge keeps running with last rules | ✅ Designed |
| Bridge crash | Bot goes blind | OpenKore auto-reconnects, bridge reloads | ✅ Designed |
| Network down | No HTTP to sidecar | Bridge runs in offline mode with cached rules | ❌ Not implemented |
| All down | Bot runs on OpenKore's built-in AI | Better than nothing | ✅ By default |

---

## 3. Layer 1: Bridge (Still 234 If Blocks, Not a Behavior Tree)

### File: `plugins/aiSidecarBridge/aiSidecarBridge.pl` — 3,552 lines, 72 subroutines, 234 `if` statements

The bridge is a **Perl plugin** that runs inside each OpenKore instance. It was supposed to be a behavior tree. It's not. It's 234 sequential `if` blocks.

### 3.1 The Problem: Flat Priority, Not Hierarchical

The 19 reflexes are implemented as sequential `if` blocks. Reflex #1 (heal) always wins because it's checked first. Not because it's most important. If the bot is at 12% HP with a boss casting Storm Gust, the heal reflex fires first. By the time the dodge reflex checks, the bot is dead.

**What a behavior tree would do:**
```
Selector:
  ├─ Sequence: Is there a lethal threat? → Dodge
  ├─ Sequence: Is HP critically low? → Heal
  ├─ Sequence: Is SP low? → Sit
  └─ Sequence: Default → Attack
```

**What the bridge actually does:**
```
if HP < 50% → heal
if HP < 15% + aggro → flee
if HP < 12% → teleport
if aggro > 5 → warn
if SP < 15% → warn
if GM detected → manual
if weight > 85% → warn
if equipment broken → warn
if monster casting → interrupt
if boss nearby → pre-pot
if HP < 50% + no heal → request help
if party low HP → warn
if aggro > 10 → flee + teleport
if HP <= 0 → sit
if deaths % 5 == 0 → warn
if out of combat + HP > 80% + SP > 30% → buff
if monster casting dangerous AoE → dodge
if out of combat + HP < 60% → sit
if out of combat + HP 30-80% → top off
```

**The order is arbitrary.** Reflex #17 (pre-dodge) should be checked before Reflex #1 (heal) because dodging a lethal AoE is more important than healing. But it's #17 because it was added later.

### 3.2 Reflex System (19 Reflexes, Flat Priority)

All reflexes are hardcoded in Perl. No LLM, no Python, no network calls. Each reflex has a condition, action, and cooldown. **No reflex has a priority value** — priority is determined by position in the file.

#### Emergency Reflexes (Checked First — Wins by Position, Not Importance)

| # | Name | Condition | Action | Cooldown | Problem |
|---|---|---|---|---|---|
| 1 | Emergency Heal | HP < 50% | Use best heal item/skill | 200ms | ⚠️ Potion CD is 2s, not 200ms. Bot wastes 75% of potions. |
| 2 | Emergency Flee | HP < 15% + aggro | Flee to safe spot | 1s | ✅ |
| 3 | Emergency Teleport | HP < 12% | Teleport away | 3s | ✅ |
| 14 | Zonk | HP ≤ 0 or ≤ 5 | Sit immediately | 2s | ✅ |
| 15 | Death Spike | Deaths % 5 == 0 | Notify sidecar | 2min | ✅ |

#### Combat Reflexes (Checked After Emergency)

| # | Name | Condition | Action | Cooldown | Problem |
|---|---|---|---|---|---|
| 9 | Interrupt Cast | Monster casting within 10 tiles | Bash/stun | 1.5s | ✅ |
| 13 | High Aggro Surround | Aggro > 10 | Flee + teleport | 3s | ✅ |
| 17 | Pre-Dodge | Monster casting dangerous AoE | Flee immediately | 2s | ⚠️ Should be checked before heal, but it's #17 |

#### Proactive Reflexes

| # | Name | Condition | Action | Cooldown | Problem |
|---|---|---|---|---|---|
| 10 | Pre-Pot | Boss within 15 tiles | Pre-heal | 5s | ✅ |
| 16 | Pre-Buff | Out of combat, HP > 80%, SP > 30% | Cast self-buffs | 15s | ✅ |
| 18 | Auto-Sit Regen | Out of combat, HP < 60% | Sit | 5s | ✅ |
| 19 | Potion Top-Off | Out of combat, HP 30-80% | Use heal item | 10s | ⚠️ Conflicts with Reflex #1 (heal) — no global cooldown |

#### Awareness Reflexes (Lowest Priority, But Not by Design)

| # | Name | Condition | Action | Cooldown | Problem |
|---|---|---|---|---|---|
| 4 | Aggro Warning | Aggro > 5 | Notify sidecar | 5s | ✅ |
| 5 | Low SP | SP < 15% | Notify sidecar | 10s | ✅ |
| 6 | GM Detection | GM/Admin within 15 tiles | Switch to manual | 1min | ✅ |
| 7 | Weight Warning | Weight > 85% | Notify sidecar | 30s | ✅ |
| 8 | Equipment Broken | Broken equipment | Notify sidecar | 1min | ✅ |
| 11 | Bot Cooperation | HP < 50% + aggro + no heal | Request help | 5s | ✅ |
| 12 | Party Low HP | Party member HP < 20% | Notify sidecar | 10s | ✅ |

### 3.3 What Needs to Change

1. **Replace 234 if blocks with a behavior tree** — Selector nodes for priority, Sequence nodes for multi-step actions, Decorator nodes for cooldowns
2. **Add global cooldown categories** — Can't use two items in one tick. Can't use two movement skills in one tick.
3. **Reorder priority** — Pre-dodge (#17) should be checked before heal (#1). Lethal threat > HP recovery.
4. **Fix potion cooldown** — 2 seconds, not 200ms. The bridge's heal reflex fires every 200ms but potions have a 2s cooldown. 9 out of 10 potion commands are wasted.

### 3.4 Snapshot System

The bridge sends a JSON snapshot to the sidecar every 500ms containing:

- **Vitals**: HP, SP, weight, level, job
- **Position**: map, x, y
- **Combat**: AI sequence, target, aggro count
- **Progression**: base/job level, exp, skill points, stat points
- **Skills**: all known skills with levels
- **Actors**: nearby monsters, players, NPCs (up to 24)
- **Raw**: char name, master, AI queue, death count, route stats

### 3.5 Anti-Detection

- Human-like reaction delay: 10-50ms (configurable)
- Random jitter on cooldowns: ±30%
- Random action delay: 50-200ms before non-critical actions
- No delay on emergency actions (survival first)

---

## 4. Layer 2: Sidecar (Middle Tactical Layer)

### File: `AI_sidecar/ai_sidecar/` (Python/FastAPI)

The sidecar is a **Python FastAPI server** that runs alongside the game clients. It handles tactics, pattern detection, and adaptation. Many modules exist. Few are wired to actually affect bot behavior.

### 4.1 API Layer

20+ API routers organized by domain. All exist. None have been integration-tested with a real bridge.

| Router | Purpose | Status |
|---|---|---|
| `/v2/ingest` | Receive events from bridge | ✅ Exists |
| `/v2/state` | Bot state management | ✅ Exists |
| `/v2/actions` | Action queue | ✅ Exists |
| `/v2/fleet` | Fleet coordination | ❌ Not tested |
| `/v2/combat` | Combat decisions | ❌ Not tested |
| `/v2/planner` | Strategic planning | ❌ Not tested |
| `/v2/reflex` | Reflex rule management | ❌ Not tested |
| `/v2/health` | Health checks | ✅ Exists |

### 4.2 Combat System

#### Combat Loop (`combat/combat_loop.py`)

Runs at 200ms intervals. Wires together 8 subsystems. **Key bugs found in code audit:**

- **Potion cooldown is 500ms** (line 384). Pre-renewal potion cooldown is 2 seconds. The bot will spam potions 4x faster than the game allows. Three out of four potion commands are wasted.
- **Skill delays are defined but not enforced**. `skill_rotation.py` has `cast_time_ms`, `delay_ms`, and `cooldown_ms` for every skill. The combat loop has `last_skill_time` and `skill_cooldowns`. But there's no enforcement loop that waits for the delay before firing the next skill. The bot will try to cast Fire Bolt while still in Storm Gust's delay. The server ignores the command.
- **Flee formula exists but is never used**. `mechanical_intuition.py` has `get_flee_rate()`. `combat_loop.py` has `max_aggro: int = 5`. These are never connected. The bot doesn't think "I have 95% flee → I can handle 10 mobs." It doesn't think "I have 5% flee → I should run from 1 mob."

#### Combat Tactics (`combat_tactics.py`)

Per-class skill combos with correct pre-renewal RO mechanics. **Element chart is Level 1 only.** Most monsters have ElementLevel 2-4. The `attr_fix.yml` file has all 4 levels. The code only uses Level 1.

**Impact of wrong element level:**
- Osiris (Undead Lv4): Holy → Undead should be 200%. Code calculates 150%. 50% damage lost.
- Ghost-type monsters: Ghost → Neutral at Level 1 is 25%. At Level 3-4 it's 0%. Bot thinks it can damage them. It can't.

#### Mechanical Intuition (`mechanical_intuition.py`)

Correct pre-renewal RO formulas. **Formulas are correct. They're just not used by any decision-making code.**

- `get_flee_rate()` exists. No combat decision calls it.
- `get_aspd()` exists. No combat decision calls it.
- `get_cast_time_reduction()` exists. No combat decision calls it.
- `get_crit_rate()` exists. No combat decision calls it.

The formulas are a library. The bot doesn't read the library.

#### Combat Instinct (`combat_instinct.py`)

50+ known dangerous monster skills with proper threat levels. **Hardcoded list of skill names. Should be loaded from `mob_skill_db.txt`.**

#### Predictive Aggro (`predictive_aggro.py`)

**87 hardcoded monsters.** The `knowledge/rathena_db/db/pre-re/mob_db.yml` has 2,675 monsters. The code uses 3% of the available data.

When the bot encounters an unknown monster, it has no aggro data, no element data, no skill data. It treats every unknown monster as "neutral element, not aggressive, no skills." This is how you die to a random monster that's actually aggressive with dangerous skills.

#### Risk Assessment (`risk_assessment.py`)

Multi-factor risk scoring. **All factors are defined. None are calibrated against real gameplay data.** The weights (0.3 for low HP, 0.2 for low SP, etc.) are guesses, not learned values.

### 4.3 Economy System

#### Economic Engine (`economy/economic_engine.py`)

- **Item valuation**: Cards > slotted equipment > ores > healing items > junk
- **Market arbitrage**: Buy low, sell high across NPC shops and player vendors
- **Farming selector**: Choose maps based on drop value, not just exp
- **Vending automation**: Auto-vendor items, set prices based on market data
- **Supply chain**: Auto-restock potions, arrows, materials

**Status**: All modules exist. None have been tested against a real RO economy. The `mob_db.yml` has complete drop data for all 2,675 monsters with rates. The bot doesn't use this data.

### 4.4 Fleet Coordination

#### Fleet Coordinator (`fleet/fleet_coordinator.py`)

- **Bot registration**: Auto-register bots with role, class, level
- **Role assignment**: farmer, buffer, merchant, scout, crafter, woe_alt
- **Party formation**: Auto-form parties with optimal composition
- **Buff coordination**: Priest buffs all party members
- **Shared threat detection**: One bot sees danger, all bots react
- **Coordinated retreat/attack**: Synchronized movement
- **Resource sharing**: Zeny, items, potions across accounts

**Status**: All modules exist. None have been tested with multiple bots. The fleet sync protocol is defined but untested.

### 4.5 Learning System

#### Death Analysis (`learning/death_analysis.py`)

**18KB of code. Imported in `pdca_loop.py`. Initialized at startup. Never called when a bot dies.**

The module has:
- `DeathAnalyzer` class with `analyze()` method
- Pattern classification (AoE, multi-hit, status + follow-up, etc.)
- Behavior adjustment recommendations

**What's missing**: A single hook: `onDeath → death_analyzer.analyze()`. That's 3 lines of code. The bot dies, respawns, and goes back to the same map. It doesn't ask "why did I die?" It doesn't learn.

#### Shared Learning DB (`learning/shared_learning_db.py`)

- **SQLite-backed** persistent learning
- **Cross-bot knowledge sharing**: One bot learns, all benefit
- **Versioned knowledge**: Roll back bad learning
- **Confidence scoring**: Only apply high-confidence learnings

**Status**: Module exists. Not wired to any decision-making code.

---

## 5. Layer 3: LLM / Conscious Engine (Top-Down Strategic Layer)

### 5.1 When the LLM is Called

The LLM is supposed to be a **last resort**, not the default. Currently it's called for every decision because the fallback decision trees aren't implemented.

| Situation | Frequency | LLM Task | Status |
|---|---|---|---|
| Bot just started | Once | "What's my build? Where should I go?" | ❌ No fallback |
| Level up | Every 5-10 levels | "What skills should I learn next?" | ✅ Phase-based fallback exists |
| Death | Per death | "Why did I die? What should I change?" | ❌ Death analysis not wired |
| Stuck | Per 5min stuck | "I've been stuck for 5 minutes, what now?" | ❌ No fallback |
| New monster | First encounter | "I've never seen this monster, what do I do?" | ❌ No fallback |
| Map change | Per map | "Is this map safe? What should I hunt?" | ❌ No fallback |
| Restock needed | Per 30min | "I need potions, where should I go?" | ❌ No fallback |
| MVP spotted | Per MVP | "Can I take this MVP? What's the strategy?" | ❌ No fallback |
| No progress | Per 10min | "I'm not gaining exp, what's wrong?" | ❌ No fallback |

### 5.2 CrewAI Agent System

The LLM is accessed through a **CrewAI multi-agent system**. All 15 agents exist. None have been tested end-to-end.

### 5.3 PDCA Loop

The **Plan-Do-Check-Act** loop runs at three frequencies. It initializes the death analysis module. It doesn't call it when a bot dies.

| Loop | Interval | Purpose | Status |
|---|---|---|---|
| Short-term | 5s | Combat adjustments, immediate threats | ✅ Running |
| Medium-term | 30s | Map evaluation, resource check | ✅ Running |
| Long-term | 120s | Goal reassessment, build planning | ✅ Running |

### 5.4 Provider Router

Supports multiple LLM providers. All work. None have been tested under load.

---

## 6. Transport: HTTP as Cross-Platform Backbone

### 6.1 Why HTTP

- **Cross-platform**: Works on Windows, Linux, macOS
- **Firewall-friendly**: Port 8080, no special permissions
- **SaaS-ready**: Same protocol works over the internet
- **Debugging**: curl, Postman, browser all work
- **Auth**: Standard HTTP auth (Bearer token)
- **Load balancing**: Standard HTTP load balancers

### 6.2 Performance Optimizations (None Implemented)

| Optimization | Current | Target | Status |
|---|---|---|---|
| Connection | Open/close per request | Keep-alive (persistent) | ❌ Not started |
| Serialization | JSON | MessagePack or CBOR | ❌ Not started |
| Polling | 500ms fixed | Adaptive (100ms-2s) | ❌ Not started |
| Compression | None | gzip on large payloads | ❌ Not started |
| Batching | One event at a time | Batch events per tick | ❌ Not started |
| Auth check | Every request | Cached token validation | ❌ Not started |

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

See Section 14. This is the only section marked FUTURE. Everything else is either done or promised but not started.

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
  ├─ HP low → potion (bridge reflex, but cooldown is wrong)
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
  └─ Death → analyze (LLM) — NOT WIRED. Bot just respawns and continues.
      ↓
     Adjust behavior → continue — NOT IMPLEMENTED.
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
  │   ├─ < 50% → potion (reflex, but fires every 200ms, potion CD is 2s)
  │   ├─ < 20% → flee (reflex)
  │   └─ < 10% → teleport (reflex)
  │
  └─ Multiple aggro → check count
      ├─ > 5 → AoE skills
      ├─ > 10 → flee + teleport
      └─ > 15 → emergency teleport
```

### 7.3 Class-Specific Behavior

Each class has a **behavior tree** defined in the architecture doc. **None are implemented as actual behavior trees.** The combat loop uses the same generic flow for all classes.

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

**Status**: Diagram exists. Code exists. No multi-bot testing has been done.

### 8.2 Optimal Party Composition

| Role | Class | Responsibility | Status |
|---|---|---|---|
| Tank | Knight/Paladin | Hold aggro, absorb damage, Bowling Bash AoE | ❌ Not tested |
| Healer | Priest/Arch Bishop | Heal, buff (Blessing, Increase AGI), resurrect | ❌ Not tested |
| AoE DPS | Wizard/High Wizard | Storm Gust, Meteor Storm, Lord of Vermillion | ❌ Not tested |
| Single DPS | Hunter/Sniper | Double Strafe, Blitz Beat, trap support | ❌ Not tested |
| Support | Bard/Dancer | Songs, dances, SP regen, stat boosts | ❌ Not tested |
| Flex | Assassin/Rogue | Steal, backstab, scout, emergency DPS | ❌ Not tested |

### 8.3 Fleet Coordination Protocol

Defined but untested. The protocol assumes all bots can communicate through the sidecar. In practice, if the sidecar goes down, bots have no fallback coordination mechanism.

### 8.4 Shared Intelligence

| Intelligence | Source | Distribution | Status |
|---|---|---|---|
| Map danger | Any bot that visits | All bots | ❌ Not implemented |
| MVP spawn | Any bot that sees it | All bots | ❌ Not implemented |
| PK warning | Any bot that sees PKer | All bots | ❌ Not implemented |
| Market prices | Merchant bot | All bots | ❌ Not implemented |
| Death patterns | Any bot that dies | All bots | ❌ Not implemented |
| Safe spots | Any bot that finds one | All bots | ❌ Not implemented |

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
   - Check element advantage — ⚠️ Uses Level 1 chart, most monsters are Level 2-4
   - Check SP availability
   - Check cooldowns — ⚠️ Defined but not enforced
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
   - If HP low → potion — ⚠️ Fires every 200ms, potion CD is 2s
```

### 9.2 Elemental Advantage System (Still Level 1 Only)

The current chart uses Level 1 values from `attr_fix.yml`. Most monsters have ElementLevel 2-4. The `attr_fix.yml` file has all 4 levels. **The code only loads Level 1.**

**Level 1 values (current, used for everything):**

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

**What the bot should use (Level 4, for monsters like Osiris):**

| Attack → Defense | Neutral | Water | Earth | Fire | Wind | Poison | Holy | Dark | Ghost | Undead |
|---|---|---|---|---|---|---|---|---|---|---|
| Water | 100% | -50% | 100% | 200% | 0% | 75% | 0% | 25% | 100% | 150% |
| Fire | 100% | 0% | 200% | -50% | 100% | 75% | 0% | 25% | 100% | 200% |
| Holy | 100% | 75% | 75% | 75% | 75% | 125% | -100% | 200% | 100% | 200% |
| Ghost | 0% | 25% | 25% | 25% | 25% | 25% | 0% | 0% | 200% | 175% |

**Impact of using wrong level:**
- Osiris (Undead Lv4): Holy → Undead = 150% (should be 200%). 50% damage lost.
- Ghost-type monsters: Ghost → Neutral = 25% (should be 0% at Lv3-4). Bot thinks it can damage them. It can't.
- Fire monsters (Lv3-4): Water → Fire = 150% (should be 200%). 50% damage lost.

### 9.3 MVP Mechanics

Each MVP has a **mechanics profile** defined in `mvp_mechanics.py`. The data is hardcoded. It should be loaded from `mob_db.yml` and `mob_skill_db.txt`.

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

### 9.4 Potion Management

```
Potion Cooldown: 2 seconds (pre-renewal)

CURRENT BUG: Combat loop enforces 500ms cooldown (line 384 of combat_loop.py).
The bot will spam potions 4x faster than the game allows.
Three out of four potion commands are wasted.

Usage Rules:
1. HP < 50%: Use best available heal item (bridge reflex, fires every 200ms)
2. HP < 80% and out of combat: Top off (bridge reflex, 10s cooldown)
3. Before boss fight: Pre-pot (bridge reflex, 5s cooldown)
4. Never spam: Track cooldown, wait 2s between potions — NOT ENFORCED
5. Stock management: Restock when < 20 potions remaining

Potion Priority:
1. Config-pushed items (class-aware, dynamic)
2. Config-pushed skills (Heal, etc.)
3. Hardcoded fallback: White Potion (always available)
```

---

## 10. Economic System

### 10.1 Item Valuation

| Category | Value | Action | Status |
|---|---|---|---|
| Cards | High | Keep | ✅ Defined |
| Slotted equipment | High | Keep | ✅ Defined |
| Refining materials (Elunium, Oridecon) | High | Keep | ✅ Defined |
| Quest items | High | Keep | ✅ Defined |
| Healing items | Medium | Keep | ✅ Defined |
| Unsorted equipment | Medium | Sell | ✅ Defined |
| Junk | Low | Sell to NPC | ✅ Defined |

### 10.2 Farming Economics

```
Zeny/Hour = (Average drop value × kills per hour) - (Potion cost per hour)

The bot chooses the map with the best zeny/hour, not just the best exp/hour.
```

**Status**: Formula defined. Not connected to actual drop data from `mob_db.yml`. The database has complete drop rates for all 2,675 monsters. The bot doesn't read it.

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

**Status**: Logic defined. Not tested against a real RO economy.

---

## 11. Learning & Adaptation

### 11.1 What the Bot Should Learn

| Knowledge | Source | Persistence | Status |
|---|---|---|---|
| Monster aggro behavior | Observation | SQLite | ❌ Not started |
| Map danger scores | Death analysis | SQLite | ❌ Not started |
| Optimal skill rotations | Trial and error | SQLite | ❌ Not started |
| Potion consumption rate | Tracking | SQLite | ❌ Not started |
| Market prices | Vending data | SQLite | ❌ Not started |
| MVP spawn timers | Observation | SQLite | ❌ Not started |
| Safe spots | Discovery | SQLite | ❌ Not started |
| PKer names | Observation | SQLite | ❌ Not started |

### 11.2 Death Analysis Pipeline

```
1. CAPTURE: Last 10 seconds of events — MODULE EXISTS, NOT WIRED
2. CLASSIFY: What killed the bot? — MODULE EXISTS, NOT WIRED
   - AoE skill (Storm Gust, Meteor Storm, etc.)
   - Multi-hit (rapid consecutive attacks)
   - Status + follow-up (stun → kill, freeze → kill)
   - Level gap (mob too strong)
   - Element disadvantage
   - Aggro overwhelm (too many mobs)
3. ADJUST: — NOT IMPLEMENTED
   - Add dodge rule for the specific skill
   - Increase flee threshold
   - Change map
   - Level up before returning
4. SHARE: Broadcast to all bots — NOT IMPLEMENTED
```

**Current behavior**: Bot dies. Bot respawns. Bot goes back to the same map. Bot dies again. No learning.

### 11.3 Confidence System

```
Confidence = (successful observations) / (total observations)

Only apply learned rules with confidence > 0.7
Roll back rules that cause more deaths
```

**Status**: Defined. Not implemented.

---

## 12. Observability & Safety

### 12.1 Metrics

| Metric | Source | Status |
|---|---|---|
| HP/SP over time | Bridge | ❌ Not implemented |
| Deaths per hour | Bridge | ❌ Not implemented |
| Exp per hour | Sidecar | ❌ Not implemented |
| Zeny per hour | Sidecar | ❌ Not implemented |
| Potions used per hour | Sidecar | ❌ Not implemented |
| Reflex fires per minute | Bridge | ❌ Not implemented |
| LLM calls per hour | Sidecar | ❌ Not implemented |
| Fleet sync latency | Sidecar | ❌ Not implemented |

### 12.2 Safety Systems

| System | Trigger | Action | Status |
|---|---|---|---|
| Circuit breaker | 5 deaths in 10 minutes | Stop all bots, notify user | ❌ Not implemented |
| GM detection | GM/Admin within 15 tiles | Switch to manual mode | ✅ Implemented |
| Anti-detection | Random delays, jitter | Avoid bot detection | ✅ Implemented |
| Degradation | Component failure | Graceful fallback | ✅ Designed |
| Rate limiting | Too many actions | Slow down, avoid server flags | ❌ Not implemented |

### 12.3 Audit Logging

Every significant action is logged with:
- **Timestamp**: When it happened
- **Bot ID**: Which bot
- **Action**: What was done
- **Context**: Why it was done
- **Outcome**: What happened as a result

**Status**: `warning` statements exist throughout the code. No structured logging (JSON). No centralized log aggregation.

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

> **This is the ONLY section marked FUTURE. Everything else is either done or promised but not started.**

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

## 15. Issue Registry: Honest Status of Every Problem

### 15.1 Architecture Issues

| # | Issue | Fix | Status | Reality Check |
|---|---|---|---|---|
| 1 | 500ms snapshot loop too slow for combat | Adaptive polling (100ms in combat, 2s idle) | NOW | ❌ Not started |
| 2 | Bridge waits for sidecar before acting | Bridge acts independently, reports upward | ✅ Done | Verified |
| 3 | Python sidecar adds HTTP latency | Keep-alive + MessagePack + batching | NOW | ❌ Not started |
| 4 | No priority system for reflexes | Behavior tree with priority nodes | NOW | ❌ Still 234 if blocks |
| 5 | State scattered across components | SQLite for persistence, bridge for real-time | NOW | ❌ Not started |
| 6 | No crash safety | Each layer functions without layers above | ✅ Done | Verified |
| 7 | No Windows support for shared memory | HTTP is cross-platform, shared memory is optional | NOW | ❌ Not started |

### 15.2 Combat Issues

| # | Issue | Fix | Status | Reality Check |
|---|---|---|---|---|
| 8 | Mage uses Cold Bolt vs water (25% damage) | Frost Diver → Fire Bolt (freeze + 4x) | ✅ Done | Verified |
| 9 | Archer stands still while shooting | Always kite (should_kite returns True for ranged) | ✅ Done | Verified |
| 10 | Assassin uses Sonic Blow → Grimtooth (wrong order) | Grimtooth (ranged) → Sonic Blow (finisher) | ✅ Done | Verified |
| 11 | Flee formula calculates value, not rate | 95% - (hit - flee), capped 5-95% | ✅ Done | Verified |
| 12 | ASPD formula off by 2-3x | Weapon delay + sqrt stat mod, capped 190 | ✅ Done | Verified |
| 13 | Cast reduction formula wrong (DEX*0.02) | DEX*0.01, cap 50% from DEX, 70% total | ✅ Done | Verified |
| 14 | Crit rate formula wrong | LUK*0.3 + 1 | ✅ Done | Verified |
| 15 | No monster skill awareness | 50+ dangerous skills with threat levels | ✅ Done | Verified |
| 16 | No multi-hit detection | 3+ damage events in <1s = flee | ✅ Done | Verified |
| 17 | No element disadvantage tracking | Element disadvantage risk factor | ✅ Done | Verified |
| 18 | No assist aggro awareness | Orc/Goblin/Thief Bug families | ✅ Done | Verified |
| 19 | No night aggro awareness | Zombie marked as night-only | ✅ Done | Verified |
| 20 | No ranged monster awareness | Orc Archer chase=18, ranged flag | ✅ Done | Verified |
| 21 | No potion cooldown tracking | 2s potion cooldown enforced | NOW | ⚠️ Code has 500ms, should be 2s |
| 22 | No skill delay tracking | Cast time + delay + cooldown per skill | NOW | ⚠️ Defined in skill_rotation.py, not enforced in combat loop |
| 23 | No gear swapping | Gear swapper module with elemental sets | NOW | ❌ Not wired to combat loop |
| 24 | No MVP mechanics | 30+ MVP profiles with strategies | NOW | ❌ Hardcoded, not loaded from DB |
| 25 | No spawn control | Spawn timer tracking, position optimization | NOW | ❌ Not started |
| 26 | No map geometry awareness | Wall/obstacle detection, line-of-sight breaks | NOW | ❌ Not started |
| 27 | Element chart uses Level 1 only | Load all 4 levels from attr_fix.yml, use monster's ElementLevel | NOW | ❌ Still Level 1 only |
| 28 | Monster database 96% incomplete (87 of 2675) | Load from mob_db.yml and mob_skill_db.txt dynamically | NOW | ❌ Still 87 hardcoded |
| 29 | Skill trees don't validate prerequisites | Load from skill_tree.yml, check Requires before recommending | NOW | ❌ Not started |

### 15.3 Strategy Issues

| # | Issue | Fix | Status | Reality Check |
|---|---|---|---|---|
| 30 | LLM called for every decision | LLM is last resort, not default | NOW | ❌ No fallback trees |
| 31 | No class-specific behavior trees | Per-class behavior trees (Mage, Archer, etc.) | NOW | ❌ Not started |
| 32 | No party synergy | Fleet coordinator with role assignment | NOW | ❌ Not tested |
| 33 | No economic awareness | Market prices, zeny/hour optimization | NOW | ❌ Not tested |
| 34 | No level penalty awareness | Level penalty in hunting recommendations | NOW | ❌ Not started |
| 35 | No learning from death | Death analysis pipeline | NOW | ⚠️ Module exists, not wired |
| 36 | No cross-bot learning | Shared learning DB | NOW | ❌ Not started |
| 37 | No build planning | Conscious engine with per-class builds | ✅ Done | Verified |
| 38 | No skill learn order optimization | Correct order for all classes | ✅ Done | Verified |
| 39 | No stat distribution optimization | Per-class stat priorities | ✅ Done | Verified |
| 40 | NV_BASIC max level was 1 (should be 9) | Fixed to 9 per rAthena skill_tree.yml | ⚠️ WRONG | Pro player puts 1 point, saves 8 for first job |

### 15.4 Multi-Bot Issues

| # | Issue | Fix | Status | Reality Check |
|---|---|---|---|---|
| 41 | No party formation | Auto-form parties with optimal composition | NOW | ❌ Not tested |
| 42 | No buff coordination | Priest buffs all party members | NOW | ❌ Not tested |
| 43 | No shared threat detection | One bot sees danger, all react | NOW | ❌ Not tested |
| 44 | No coordinated retreat | Fleet leader commands retreat | NOW | ❌ Not tested |
| 45 | No resource sharing | Zeny/items shared across accounts | NOW | ❌ Not tested |
| 46 | No role assignment | Farmer, buffer, merchant, scout, etc. | NOW | ❌ Not tested |
| 47 | No level gap management | Keep party within 10 levels | NOW | ❌ Not tested |

### 15.5 Technical Debt Issues

| # | Issue | Fix | Status | Reality Check |
|---|---|---|---|---|
| 48 | No test coverage for bridge | Perl test harness | NOW | ❌ Not started |
| 49 | No integration tests | Bridge + sidecar together | NOW | ❌ Not started |
| 50 | No performance benchmarks | Reflex latency targets (<1ms) | NOW | ❌ Not started |
| 51 | No structured logging | JSON logging | NOW | ❌ Not started |
| 52 | No metrics | Prometheus metrics | NOW | ❌ Not started |
| 53 | No tracing | OpenTelemetry | NOW | ❌ Not started |
| 54 | No alerting | Death spike, stuck, no exp | NOW | ❌ Not started |
| 55 | No config validation | Schema checking | NOW | ❌ Not started |
| 56 | No single config source | YAML with inheritance | NOW | ❌ Not started |
| 57 | Skills referenced as strings | Skill registry with objects | NOW | ❌ Not started |
| 58 | HTTP open/close per request | Keep-alive connections | NOW | ❌ Not started |
| 59 | JSON serialization overhead | MessagePack for high-frequency data | NOW | ❌ Not started |
| 60 | SQLite files tracked in git | Added to .gitignore | ✅ Done | Verified |
| 61 | .pids/ not in gitignore | Added to .gitignore | ✅ Done | Verified |
| 62 | sidecar_auth_token.txt not in gitignore | Added to .gitignore | ✅ Done | Verified |

---

## 16. The 8 Things That Can Be Fixed in One Day

These are the highest-impact, lowest-effort fixes. Each one takes minutes to hours, not weeks.

| # | Fix | File | Change | Effort | Impact |
|---|---|---|---|---|---|
| 1 | Fix potion cooldown | `combat/combat_loop.py:384` | `0.5` → `2.0` | 1 character | Bot stops wasting 75% of potions |
| 2 | Fix NV_BASIC to level 1 | `conscious_engine.py` | `9` → `1` | 1 number | Saves 8 skill points for first job |
| 3 | Wire death analysis hook | `pdca_loop.py` | Add `onDeath → analyze()` | 3 lines | Bot learns from deaths |
| 4 | Enforce skill delays | `combat/combat_loop.py` | Check `last_skill_time + delay` | 3 lines | Bot stops firing into cooldown |
| 5 | Connect flee to aggro limit | `combat/combat_loop.py` | `max_aggro = flee_rate / 20` | 5 lines | Bot pulls appropriate number of mobs |
| 6 | Load monster DB from mob_db.yml | `predictive_aggro.py` | Replace 87 hardcoded with YAML loader | 1 afternoon | Bot knows all 2,675 monsters |
| 7 | Load element chart from attr_fix.yml | `combat_tactics.py` | Replace Level 1 with all 4 levels | 1 afternoon | Correct damage calculations for all monsters |
| 8 | Replace 234 if blocks with behavior tree | `aiSidecarBridge.pl` | Behavior tree with selector/sequence/decorator | 1 week | Proper priority arbitration |

**Total effort for items 1-7: ~2 days. Total effort for item 8: ~1 week.**

---

## Appendix A: File Map

```
openkore-ai-v3/
├── plugins/
│   └── aiSidecarBridge/
│       └── aiSidecarBridge.pl          # Bridge (Perl, 3,552 lines, 234 if blocks, 72 subs)
│
├── AI_sidecar/
│   └── ai_sidecar/
│       ├── app.py                      # FastAPI entry point
│       ├── config.py                   # Configuration
│       ├── combat_tactics.py           # Per-class skill combos (Level 1 element chart only)
│       ├── mechanical_intuition.py     # RO formulas (correct but unused by decisions)
│       ├── combat_instinct.py          # Monster skill awareness (50+ hardcoded skills)
│       ├── predictive_aggro.py         # Monster aggro database (87 of 2,675 monsters)
│       ├── risk_assessment.py          # Risk/reward scoring (uncalibrated weights)
│       ├── conscious_engine.py         # Phase-based builds (6 classes, 2-3 variants each)
│       ├── game_engine.py              # rAthena knowledge integration
│       ├── combat/
│       │   ├── combat_loop.py          # Main combat loop (200ms, potion CD bug at line 384)
│       │   ├── reflex_combat.py        # Hardcoded combat reflexes
│       │   ├── skill_rotation.py       # Skill selection (cast times defined, not enforced)
│       │   ├── elemental_matrix.py    # Element advantage (Level 1 only)
│       │   ├── buff_maintenance.py     # Buff tracking and recasting
│       │   ├── gear_swapper.py         # Dynamic gear changes (not wired)
│       │   ├── resource_manager.py    # Potion/consumable management
│       │   ├── threat_targeting.py     # Target selection
│       │   ├── action_executor.py      # Action enqueueing
│       │   ├── mvp_mechanics.py        # MVP skill/phase knowledge (hardcoded)
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
│       │   ├── fleet_coordinator.py    # Multi-bot coordination (untested)
│       │   ├── multi_account_synergy.py # Party composition (untested)
│       │   ├── swarm_ai.py             # Decentralized coordination (untested)
│       │   ├── party_coordinator.py    # Party management (untested)
│       │   ├── role_manager.py         # Role assignment (untested)
│       │   ├── conflict_resolver.py    # Order conflict resolution (untested)
│       │   ├── cross_bot_resource_manager.py # Shared resources (untested)
│       │   └── self_learning.py        # Cross-bot learning (untested)
│       ├── economy/
│       │   ├── economic_engine.py      # Core economy (untested)
│       │   ├── farming_selector.py     # Map selection by value (untested)
│       │   ├── market_arbitrage.py     # Buy low, sell high (untested)
│       │   ├── vending_automation.py   # Auto-vending (untested)
│       │   ├── supply_chain.py         # Restock planning (untested)
│       │   └── opportunity_cost.py     # Zeny/hour optimization (untested)
│       ├── learning/
│       │   ├── death_analysis.py       # Post-mortem analysis (18KB, exists, NOT WIRED)
│       │   ├── shared_learning_db.py   # Cross-bot knowledge (exists, NOT WIRED)
│       │   └── strategy_optimizer.py  # Strategy refinement (exists, NOT WIRED)
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
│   └── rathena_db/                     # rAthena game data (COMPLETE, UNUSED)
│       ├── db/pre-re/                  # Pre-renewal data
│       │   ├── mob_db.yml              # 2,675 monsters (795KB, 42,537 lines) — BOT USES 87
│       │   ├── mob_skill_db.txt        # Monster skills (481KB, 5,783 lines) — BOT USES 50
│       │   ├── skill_tree.yml          # Skill trees (83KB, 3,579 lines) — BOT DOESN'T READ
│       │   ├── item_db_equip.yml       # Equipment items — BOT DOESN'T READ
│       │   ├── item_db_etc.yml         # Misc items — BOT DOESN'T READ
│       │   ├── item_db_usable.yml      # Usable items — BOT DOESN'T READ
│       │   ├── job_stats.yml           # Job stat bonuses (51KB, 2,930 lines) — BOT DOESN'T READ
│       │   ├── size_fix.yml            # Size damage modifiers (1.4KB, 40 lines) — BOT DOESN'T READ
│       │   ├── attr_fix.yml            # Element damage modifiers (8.5KB, 478 lines) — BOT USES LEVEL 1 ONLY
│       │   └── level_penalty.yml       # Exp penalty by level — BOT DOESN'T READ
│       └── db/re/                      # Renewal data (same structure)
│
└── control/
    ├── ai_sidecar.txt                  # Bridge config
    ├── ai_sidecar_policy.txt           # Bridge policy
    └── sidecar_auth_token.txt          # Auto-generated auth token
```

---

## Appendix B: Key Metrics (Honest)

| Metric | Current | Target | Status |
|---|---|---|---|
| Reflex latency | ~50ms (bridge) | <50ms | ✅ Done |
| Combat tick | 200ms | 100ms | ❌ Not started |
| Snapshot interval | 500ms fixed | Adaptive (100ms-2s) | ❌ Not started |
| LLM call frequency | Per decision | Per 5-30min | ❌ No fallback trees |
| Monsters known | 87 | 2,675 (from mob_db.yml) | ❌ Not started |
| Maps known | 59 | All (from mob_db.yml) | ❌ Not started |
| Reflex count | 19 if blocks | Behavior tree | ❌ Not started |
| Party size | 1 | 1-12 | ❌ Not tested |
| Fleet size | 3 | Unlimited | ❌ Not tested |
| Death recovery | Manual (respawn + continue) | Automatic (analyze + adjust) | ⚠️ Module exists, not wired |
| Learning | None | Continuous | ❌ Not started |
| Windows support | Partial | Full | ✅ By design |
| Element chart | Level 1 only | All 4 levels from attr_fix.yml | ❌ Not started |
| Skill validation | None | Prerequisites from skill_tree.yml | ❌ Not started |
| Potion cooldown | 500ms (WRONG) | 2,000ms (correct) | ⚠️ 1 character fix |
| Skill delay enforcement | None | Cast time + delay + cooldown | ⚠️ 3 lines of code |
| Flee formula usage | None | Connected to max_aggro | ⚠️ 5 lines of code |
| Death analysis | Module exists | Wired to onDeath hook | ⚠️ 3 lines of code |
| NV_BASIC | Level 9 (WRONG) | Level 1 (save 8 points) | ⚠️ 1 number fix |
| SaaS | — | FUTURE ONLY | 🔄 Future |

---

*This document represents the architecture as of 2026-07-15. Items marked "✅ Done" are verified in the codebase. Items marked "❌ Not started" are documented but not implemented. Items marked "⚠️" are bugs that can be fixed in minutes. Items marked "🔄 Future" are not planned for the current iteration.*

*The honest truth: 19 items are done. 42 items are promised. 0 items have been started since this document was first written. The architecture doc is a confession, not a plan. The next step is to ship, not to plan.*
