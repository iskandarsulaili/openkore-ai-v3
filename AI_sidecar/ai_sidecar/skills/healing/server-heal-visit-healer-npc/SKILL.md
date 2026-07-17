---
name: server-heal-visit_healer_npc
description: Server healing strategy
version: 1.0.0
triggers:
  - visit_healer_npc_requested
  - low_hp
when_to_use:
  - hp_ratio < 0.30
  - strategy == visit_healer_npc
metadata:
  domain: healing
  source: crewai_discovery_agent
  confidence: 0.85
  server_map: prontera
  target_npc: Healer#prt
---

# Discovered Heal Strategy: visit_healer_npc

- **Command**: move 159 193
- **Target**: prontera
- **NPC**: Healer#prt
- **Confidence**: 0.85
