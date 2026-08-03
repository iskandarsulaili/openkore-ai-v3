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
                     │  Slow, deliberate, pre-emptive. Solves novel  │
                     │  situations: unknown maps, quest chains, NPC/ │
                     │  player interaction, cold-start strategy,     │
                     │  delegating across the fleet. Updates reflex  │
                     │  rules + trains the subconscious.             │
                     └───────────────────────┬───────────────────────┘
                                             │ feedback (success/failure)
                     ┌───────────────────────▼───────────────────────┐
                     │  SUBCONSCIOUS — trained ML from conscious action│
                     │  Unsupervised punish/reward loop: learns which  │
                     │  behaviors win. Speeds up decisions that the    │
                     │  conscious already figured out, so they become  │
                     │  automatic and don't need re-reasoning.         │
                     └───────────────────────┬───────────────────────┘
                                             │ compiled rules
                     ┌───────────────────────▼───────────────────────┐
                     │  REFLEX — basic/chain/complex rule set          │
                     │  Instant, zero-latency: skill combos, opponent  │
                     │  skill cancelling, quick dodging, potion, sit.  │
                     │  Updated by the conscious over time.            │
                     └─────────────────────────────────────────────────┘
```

**The three tiers are NOT separate bots.** They are one decision pipeline. A novel situation is reasoned by the conscious; if it recurs, the subconscious learns it; if it becomes repetitive and needs instant reaction, it is compiled into a reflex. This is exactly how a human expert plays: deliberate at first, automatic at mastery.

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
| Phase 1 — map-agnostic routing | ⬜ pending | route via discovered portal graph |
| Phase 2 — conscious dispatch | ⬜ pending | unified_consciousness + agents + LLM |
| Phase 3 — subconscious RL | ⬜ pending | reward/punish by outcome |
| Phase 4 — verify + server-agnostic proof | ⬜ pending | tests + live fleet |

_Update this tracker at the end of each phase. A phase is "done" only when its verification passes and the tracker is updated in the same commit._
