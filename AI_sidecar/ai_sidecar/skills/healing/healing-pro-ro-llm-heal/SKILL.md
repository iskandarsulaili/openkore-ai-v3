---
name: healing-pro_ro_llm-heal
description: "pro_ro_llm discovered strategy for healing/heal"
version: 1.0.0
triggers:
  - heal_strategy
  - heal
when_to_use:
  - action == heal
  - domain == healing
metadata:
  domain: healing
  source: post_action_review
  agent: pro_ro_llm
  confidence: 0.6
---

# Healing Strategy: Heal

## Discovered by pro_ro_llm

- **strategy**: visit_healer_npc
- **target_map**: prontera
- **target_npc**: Healer#prt
- **confidence**: 0.85
- **discovered_by**: test_bot

## Context at discovery
- **Map**: prontera
- **HP**: ?/?
- **Zeny**: ?
- **Level**: ?
