---
name: subconscious-driven-combat
description: "Moment-to-moment combat is driven by trained ML (subconscious), not rule-coded combos."
version: 1.0.0
triggers:
  - combat
  - attack
  - skill_combo
  - dodge
when_to_use:
  - engaged in combat / farming a mob
  - choosing skill combos, potting beats, target cadence
when_not_to_use:
  - novel/strategic situations (escalate to conscious tier)
  - lethal-threat safety (reflex floor)
metadata:
  domain: combat
  subdomain: subconscious
  source: reinforcement_learner
  confidence: 0.9
  tags: [combat, ml, dqn, subconscious]
---
# Subconscious-Driven Combat

## Core Rule
~95% of moment-to-moment skilled combat is DRIVEN by the **trained ML subconscious**
(`learning/reinforcement_learner.py`, `ml_subconscious/`), promoted from shadow via a
reward/punish loop. It is trained muscle memory — automatic because TRAINED, not because
rule-coded.

## Three-Tier Split
- **Subconscious** = where the Pro's SPEED lives: potting at the right beat, target
  cadence, route-feel, combo chains. DRIVES the majority of combat.
- **Reflex** = hardwired safety floor ONLY (never-die, withdraw, flinch). Immutable,
  non-learned. NOT 'compiled mastery'.
- **Conscious** = strategic intent + whole-picture root-cause for novel situations. Does
  NOT micro-manage every per-cycle action.

## How
1. Subconscious emits skilled combat actions from its trained policy.
2. Reflex overrides instantly on lethal threat.
3. Conscious steps in for novel situations, then re-trains the subconscious via the
   reward/punish loop (promote from shadow on verified success).

## Verification
Real DQN entry = `._train_from_replay`. Verify via `data/reinforcement_stats.json`
(training_steps > 0), not code reads.
