---
name: server-heal-strategy
description: "Optimal healing strategy for this RO server based on discovered NPCs"
version: 1.0.0
triggers:
  - low_hp
  - heal_strategy_requested
when_to_use:
  - hp_ratio < 0.30
when_not_to_use:
  - bot has potions and is in a dungeon (use potion not NPC)
metadata:
  domain: healing
  subdomain: npc_interaction
  source: crewai_discovery_agent
  confidence: 0.85
  tags: [healing, npc, prontera]
---
# Server Healing Strategy

## Discovered Healer NPC
- **Map**: Prontera
- **Coordinates**: (159, 193)
- **NPC Name**: Healer#prt
- **Type**: Free full heal
- **Discovered by**: CrewAI discovery agent from OpenKore tables (npcs.txt)

## How to Use
1. Move to coordinates 159 193 on Prontera map
2. Talk to the NPC: `talknpc 159 193 c r0 n`
3. The NPC provides free full HP recovery

## When to Use
- Bot HP is below 30%
- Bot is on Prontera map (safe town)
- No healing potions available

## When NOT to Use
- Bot is in a dungeon (walking back to town wastes time)
- Bot has sufficient potions (use potions instead)
- Bot HP is above 50% (continue grinding instead)
