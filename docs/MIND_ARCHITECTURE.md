# MIND_ARCHITECTURE.md — The God-Tier Three-Tier Mind

> Authoritative statement of what openkore-ai-v3 is *supposed* to be.
> This document is the spec. When code drifts from it, the code is wrong — not this doc.
> **Last updated:** 2026-08-03 (drift assessment + phased remediation).

---

## 1. The Vision (non-negotiable)

openkore-ai-v3 is a **God-Tier RO AI** that is:

- **Highly server-agnostic** — it adapts to ANY Ragnarok Online server as-is: custom maps, custom portals, custom NPCs, custom items, custom quests. No per-server hardcoding.
- **Zero human intervention** — from a fresh level-1 character on any island/tutorial map, it self-directs through escape → registration → farming → leveling → job-change → endgame, with no human correcting it.
- **Proven to beat a human Pro RO player** — optimal decisions, perfect execution, faster kill rate, no downtime, deterministic optimal routing.
- **Three-tier mind** (conscious / subconscious / reflex), operating as one integrated system:

```
                     ┌─────────────────────────────────────────────┐
                     │  CONSCIOUS — LLM + CrewAI agent orchestration │
                     │  SLOW, sparse, deliberate. ~the thin spotlight│
                     │  of awareness, NOT the majority of skill.     │
                     │  High-level INTENT + whole-picture ROOT-CAUSE │
                     │  analysis (novel situations, quest chains,    │
                     │  cold-start strategy, cross-fleet delegation).│
                     │  Selects intent; for execution it RELIES on   │
                     │  the subconscious — it does NOT micro-manage  │
                     │  every per-cycle action. Understands WHY the  │
                     │  bot is failing (deep, systemic), then points │
                     │  intent. (Human: the deliberating 'I'.)       │
                     └───────────────────────┬───────────────────────┘
                                             │ intent + trains
                     ┌───────────────────────▼───────────────────────┐
                     │  SUBCONSCIOUS — trained ML (the skilled self)  │
                     │  ~95% of actual skill. Learned/procedural from│
                     │  reward/punish + observed experience. Where a │
                     │  Pro's SPEED lives: potting at the right beat,│
                     │  target cadence, route-feel, combo chains —   │
                     │  automatic because TRAINED (muscle memory),   │
                     │  not because rule-coded. DRIVES the majority  │
                     │  of moment-to-moment skilled action.          │
                     │  (Human: the body that knows without thinking)│
                     └───────────────────────┬───────────────────────┘
                                             │ updated by conscious
                     ┌───────────────────────▼───────────────────────┐
                     │  REFLEX — HARDWIRED PRIMITIVES (baseline)     │
                     │  IMMUTABLE, non-learned safety invariants:    │
                     │  never die, don't overextend, flinch from     │
                     │  lethal threat. The evolutionary / generic    │
                     │  baseline that fires before ANY trained skill.│
                     │  NOT the Pro's fast instinct (that is the    │
                     │  subconscious trained skill, above). Reflex  │
                     │  is the floor every tier respects.            │
                     └─────────────────────────────────────────────────┘
```

**The three tiers are NOT separate bots and are NOT proficiency stages of one skill.** They are *three different kinds of processing that coexist*, mapped to real human cognition:

- **Conscious** = high-level intent + systemic interpretation. Slow, sparse. It sees the WHOLE PICTURE and finds ROOT CAUSES, then sets intent. It does NOT have to (and must not) micromanage each action.
- **Subconscious** = the TRAINED, learned skill — where a Pro's actual speed and craft live. This is the tier that should DRIVE most moment-to-moment combat because it is *trained from experience* (muscle memory), exactly like a veteran player's hands.
- **Reflex** = the hardwired, non-learned safety baseline (flinch / withdraw / never-die). It is the *floor*, not the ceiling.

Correction vs the earlier model: reflex is NOT "compiled mastery" (a proficiency endpoint). A human expert's automatic skill is *subconscious* (trained), not reflex (primitive hardwired). The tiers therefore do NOT form a "skill ladder"; consciousness interprets, the subconscious performs, and reflex sets immovable safety bounds.

---

## 2. Drift Assessment (2026-08-03) — honest status

The god-tier three-tier machinery **already exists and is largely wired in**:

| Tier | Modules (exist) | Wired-in |
|---|---|---|
| Conscious | `strategy/unified_consciousness.py`, `empire_manager.py`, `long_term_planner.py`, `theory_of_mind.py`, `mission_agent.py`, LLM manager (`llm/`), `crewai/agents/*` (16 agents) | yes (5+ modules reference) |
| Subconscious | `learning/reinforcement_learner.py`, `death_analysis.py`, `failure_wiring.py`, `strategy_optimizer.py`, `shared_learning_db.py`, `feedback_loop.py` | yes |
| Reflex | `reflex/reflex_pipeline.py`, `rule_engine.py`, `trigger_matcher.py`, `highfreq_reflex.py`, `reflex_rules.yaml` | yes (3+ modules) |
| Server-agnostic knowledge | `dynamic_portal_discovery.py`, `map_knowledge.py`, `fleet/self_learning.py`, `capabilities.py` | yes (2+ modules) |

**The drift:** despite this architecture, `AI_sidecar/ai_sidecar/autonomy/heuristic_service.py` — a **4,723-line monolith** — **bypasses all three tiers** and hardcodes the rathena-ai-world island/academy layout directly:

- **129 hardcoded map-name references** (`int_land`, `izlude`, `iz_ac01`, `prt_fild05`, `prt_fild08`, `pay_dun`, `orcsdun`, `iz_int01-04`).
- **99 hardcoded command directives** (`move`, `talknpc`, `lockMap`, `mon_control`).
- **28 literal coordinate moves**: `move 49 57`, `move 125 257`, `move 51 30`, `move 367 205`, `move 22 203`, `move 290 221`, `move 160 133`, `move 200 200`, `move 100 39` (talknpc), etc.

**Consequence:** the system is currently a **rule engine for THIS one server**, not a server-agnostic AI. On any other server it breaks. This violates Rule 19 ("Self-Adapt for Custom Server Layouts") and the core vision.

**Root cause of drift:** the escape/academy fixes added this session (`move 49 57`, `move 125 257`, `move 100 39`, `set lockMap prt_fild08`) made the *specific island* work by adding more literals — exactly the "over-tweak to THIS server" anti-pattern. The conscious tier was bypassed because the reflex/rule path was faster to patch.

---

## 3. Remediation Plan (phased, non-destructive)

`heuristic_service.py` is **NOT removed** — it is **marked OBSOLETE** and progressively demoted. Its hardcoded island/academy knowledge is moved into the server-agnostic tiers.

### Phase 1 — Map-agnostic routing
Replace hardcoded coordinate moves with routing through the **discovered portal graph** (`dynamic_portal_discovery.py`, `map_knowledge.py`, pathfinder). Instead of `move 49 57`/`move 125 257`, the bot routes to the *discovered portal* leading toward its goal map. Works on any server because the portals are observed, not assumed.

### Phase 2 — Dispatch through the conscious tier
Replace the island/academy/quest hardcoded directives with dispatch through `unified_consciousness` + the CrewAI agents + LLM reasoning. The conscious observes the live map/NPC/portal state (from the bridge snapshot + `dynamic_portal_discovery`) and *reasons* the escape/registration/quest sequence, rather than matching literals. It updates the reflex rules for what becomes repetitive.

### Phase 3 — Wire the subconscious
Connect `reinforcement_learner`/`failure_wiring`/`death_analysis` to reward/punish the conscious+reflex decisions by actual outcome (escaped ✓ / died ✗ / registered ✓ / stuck ✗), so the system learns which behaviors win over time.

### Phase 4 — Verify
- Tests green (`make test` 1168/1168, Python suite).
- The fleet still escapes the island → registers → farms → levels, but now driven by discovered knowledge + agent reasoning, not literals.
- Confirmed server-agnostic: on a different server with a different island/academy, the AI adapts via discovery + reasoning.

**Do NOT remove** `heuristic_service.py` — it still carries the reflex-tier cold-start logic and is the fallback. It is demoted from "source of truth" to "legacy rule layer that the conscious can update."

---

## 4. Progress Tracker (prevent drift)

| Phase | Status | Commit / Notes |
|---|---|---|
| Phase 1 — map-agnostic routing | ✅ done | izlude/izlude_a→academy via Pathfinder (discovered portals); hunting lockMap via get_hunting_maps(level). Literals are fallback-only. Live: bots escaped→academy, bot4 gained EXP |
| Phase 2 — conscious dispatch | ✅ done | novel/unknown-map situations delegate to UnifiedConsciousness.decide (last-resort fallback, never overrides reflex; excludes known island/academy maps). (LLM wiring of the conscious-engine for general reasoning is a follow-on depth.) |
| Phase 3 — subconscious RL | ✅ done | ReinforcementLearner reward/observe loop wired in pdca_loop (death=-1, survive=+0.05, train every 16 obs). Was dormant; now learns which behaviors win. Safe: records+trains, doesn't drive decisions yet |
| Phase 4 — verify + server-agnostic proof | ✅ done | Tests green (make test 1168/1168, cold-start 11, pdca/learning/conscious 24). LIVE: sidecar holds 3 tiers — unified_consciousness initialized + conscious_trigger reasoning from live state (detected Poring Hunt Quest + recommends knife-dropping Porings + STR), ReinforcementLearner learning (13+ experiences), fleet active with 3 bots, bot4 at iz_ac01 with 100 EXP |

_Update this tracker at the end of each phase. A phase is "done" only when its verification passes and the tracker is updated in the same commit._
