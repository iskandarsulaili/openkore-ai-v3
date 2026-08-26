---
name: agent-driven-gear
description: "Gear/consumable/equipment decisions must come from LLM/CrewAI, never hardcoded item IDs."
version: 1.0.0
triggers:
  - gear
  - equipment
  - potion
  - restock
  - weapon
when_to_use:
  - deciding what gear/consumables to carry, equip, restock, or upgrade
  - low sustain while farming
when_not_to_use:
  - reflex instant-timing actions (potting on a beat, dodging)
metadata:
  domain: economy
  subdomain: gear_progression
  source: gear_progression_planner
  confidence: 0.95
  tags: [gear, llm, server-agnostic]
---
# Agent-Driven Gear Decisions

## Core Rule
Whether a bot carries a weapon, potions, armor, or any equipment — and how it acquires,
equips, restocks, or upgrades them — MUST be decided by the **LLM and/or CrewAI agents**
(conscious tier), never hardcoded if/else rules or baked item IDs.

## Why
Gear needs are server-agnostic and situation-dependent. A fresh server may hand out a
different starter kit; a Pro player adapts gear to the map/opponent. The conscious tier
reasons "I lack sustain to kill X → acquire Y" from live observation; the subconscious RL
learns which gear decisions lead to kills/rewards.

## How
1. Observe live state (HP/SP/zeny/map/mob) from the snapshot.
2. Conscious tier decides WHAT gear is needed and WHY.
3. Executor translates using the DB-backed `server_solutions` store (which potion item to
   buy on THIS server — learned, never literal).
4. Reflex may only ACT on that decision with instant timing.

## Pitfall
A hardcoded per-server gear rule (`buy 501`, `set attackAuto 3`) is a BUG to replace with
agent reasoning, not a feature.
