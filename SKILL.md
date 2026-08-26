---
name: openkore-ai-v3
description: "Operate/develop the openkore-ai-v3 God-Tier RO bot: three-tier mind, server-agnostic, self-learning/healing/improving loop."
version: 1.0.0
author: Hermes Agent + lot399
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [openkore, ragnarok, bot, three-tier-mind, self-improving, server-agnostic, sidecar]
    related_skills: [openkore-ai-system, pro-ro-bot-development, ro-game-mechanics, deep-audit-pattern]
    homepage: https://github.com/openkore/openkore
---

# openkore-ai-v3 — God-Tier Server-Agnostic RO Bot

openkore-ai-v3 is a **three-tier-mind** Ragnarok Online farming bot that must be
**server-agnostic** and **zero-human-intervention**: from a fresh level-1 character it
self-directs escape → registration → farming → leveling → job-change → endgame on ANY
server, beating a human Pro player, with no human correcting it.

This SKILL.md is the operational contract for an agent working in this repo. It is the
**Hermes-facing** companion to the bot's own internal self-* system
(`AI_sidecar/ai_sidecar/memory/{SOUL,MEMORY}.md` + `skills/` + `self_awareness.py`). Both
follow the same Hermes pattern: curated identity + curated durable lessons, injected into
every reasoning call, with the agent writing lessons back when it learns something.

## The Three-Tier Mind (non-negotiable)

| Tier | Role | Where it lives |
|---|---|---|
| **Conscious** | High-level intent + whole-picture ROOT-CAUSE. Solves novel situations, cold-start, cross-fleet delegation. Sets intent, does NOT micro-manage per-cycle. | `strategy/unified_consciousness.py`, `crewai/agents/*`, `llm/`, `empire_manager.py`, `autonomy/pdca_loop.py` |
| **Subconscious** | ~95% of moment-to-moment skilled combat. TRAINED ML (muscle memory), promoted from shadow via reward/punish. | `learning/reinforcement_learner.py`, `ml_subconscious/`, `combat/` |
| **Reflex** | HARDWIRED safety floor ONLY (never-die, withdraw, flinch). Not strategy, not 'compiled mastery'. | `reflex/reflex_pipeline.py`, `reflex_rules.yaml` |

The tiers are **not separate bots and not proficiency stages** — they are three kinds of
processing that coexist. Reflex is the floor every tier respects; consciousness interprets,
the subconscious performs.

## Hard Rules (RULE.md — authoritative)

These are binding. Violating them is a BUG, not a feature.

1. **`autonomy/heuristic_service.py` is OBSOLETE.** Do NOT remove it (carries reflex
   cold-start + fallback), do NOT add new hardcoded map/coord/NPC directives to it. Routing
   goes through the **discovered portal graph** (`dynamic_portal_discovery.py`,
   `map_knowledge.py`, pathfinder) + the conscious tier + LLM — never literals like
   `move 49 57` / `move 125 257` / `talknpc 100 39`.
2. **Gear/consumable/equipment decisions are AGENT-DRIVEN, never hardcoded.** What a bot
   carries, equips, restocks, upgrades — decided by LLM/CrewAI, not if/else or baked item
   IDs. Reflex may only ACT on a conscious decision with instant timing.
3. **Server-specific facts live in the DB, NEVER in `*.py`.** No `buy 501`, no
   `move prontera`, no `prt_fild08c`, no `mon_control Thief Bug` literals. Learned +
   persisted in the DB-backed `server_solutions` store, filled by observing the live server.
   LLM/CrewAI decides WHAT; the executor translates using learned facts.
4. **Self-* autonomy.** Every novel/agnostic situation resolves via CrewAI/LLM reasoning from
   live observation — never a per-server hardcoded rule. If a rule is 'hardcoded for one
   server' it's a BUG.
5. **Zero mocks/stubs/dormant code. Reconcile, never trim.** A dead code path may be an
   incomplete impl that's still needed — dig deeper before assuming.
6. **When a fork construct misbehaves, MATCH a proven stock example verbatim** before
   guessing semantics.

## Self-Learning / Self-Healing / Self-Improving Loop

The bot is self-LEARNING, self-HEALING, self-IMPROVING through a Hermes-pattern loop:

```
Live observation (bridge snapshot / DB / portals.txt)
   → Conscious LLM reasons (SOUL + MEMORY injected every call)
   → Action emitted → executed → acked (action_queue.completed_actions)
   → post_action_review.record_lesson() writes a MEMORY.md lesson on
     learning-worthy outcomes (fail/refuse/error)   [wired at pdca_loop.py ~5469]
   → MEMORY.md re-injected into the next reasoning call
   → skills_manager creates/updates a SKILL.md from a verified discovery
```

**Where the pieces live:**

- `AI_sidecar/ai_sidecar/memory/SOUL.md` — curated identity + decision doctrine. Injected
  VERBATIM into every conscious-tier LLM call.
- `AI_sidecar/ai_sidecar/memory/MEMORY.md` — curated durable lessons, char-bounded
  (100,000), `\n§\n`-delimited. The conscious LLM writes lessons when it decides something
  is worth remembering. This is NOT a DB dump.
- `AI_sidecar/ai_sidecar/memory/self_awareness.py` — the SelfAwareness layer. `inject()`
  prepends SOUL+MEMORY to every call (wired into `llm/manager.py` + `model_router.py`).
  `add_lesson()` appends a lesson; P2P crowdsource sink pushes/pulls shared lessons.
- `AI_sidecar/ai_sidecar/memory/lessons_hub.db` — fleet-shared central sink (SQLite).
- `AI_sidecar/ai_sidecar/skills/` — the sidecar's OWN skill library (manager/loader/usage/
  curator, all Hermes-inspired). `skills_loader.load_for_context()` progressive-disclosure.
- `AI_sidecar/ai_sidecar/autonomy/post_action_review.py` — `record_lesson()` writes
  MEMORY.md lessons; `review_action()` / `review_heal_strategy()` create skills from
  discoveries.
- `AI_sidecar/ai_sidecar/autonomy/pdca_loop.py` — the PDCA loop. ~line 5469: completed
  actions → long_term_memory (DB) + a curated MEMORY.md lesson on failure/refusal.

**Agent discipline in this repo:** when you fix something or learn a durable lesson, write
it back — either a `record_lesson` call in the relevant code path or (if a Hermes agent
working on the repo) add a lesson to `AI_sidecar/ai_sidecar/memory/MEMORY.md` and/or a
`references/` file + pointer in this SKILL.md. Honor lessons already in MEMORY.md — never
repeat a known mistake.

## Key Architecture

- **Sidecar** = `AI_sidecar/ai_sidecar/` (Python, FastAPI). Handles ALL decisions:
  config, routing, economy, combat, gear, progression. Entry: `app.py`, build:
  `lifecycle.py` (`RuntimeState`).
- **Bridge** = `plugins/aiSidecarBridge/aiSidecarBridge.pl` (Perl). ONLY for:
  1. monitor/report state (snapshots), 2. pre-command interception (block/redirect
  `move prontera`), 3. portal-exit reflex, 4. portal-route redirect. Passes commands; never
  the source of strategic/tactical decisions.
- **PDCA Loop** = `autonomy/pdca_loop.py` (large, ~10.7k lines). Emits heuristic /
  game-engine / swarm / vendor / skill / combat actions into the action queue; ack results
  come back via `action_queue.completed_actions`.
- **Conscious agents** = `crewai/agents/*` (16 agents). `maybe_create_skill()` lets any
  agent persist a discovery as a skill.
- **Subconscious** = `learning/reinforcement_learner.py` + `ml_subconscious/` (DQN).
- **Reflex** = `reflex/` + `reflex_rules.yaml`.
- **Server-agnostic knowledge** = `dynamic_portal_discovery.py`, `map_knowledge.py`,
  `fleet/self_learning.py`, `capabilities.py`, `server_solutions` store.

## Common Pitfalls

- **Hardcoding server literals** — the #1 anti-pattern. Always resolve data-driven from the
  live portal graph + server_solutions store, never a coord/map/item literal in `*.py`.
- **Over-tweaking heuristic_service.py** to make one island work — adds literals, breaks
  server-agnosticism. Route through discovery + conscious tier instead.
- **Treating reflex as 'compiled mastery'** — it's the immutable safety floor, not trained
  skill. Subconscious is where trained skill lives.
- **Editing pdca_loop.py blindly** — it's huge; the completed-action hook is around
  ~line 5469; use targeted patches, not full rewrites.
- **MEMORY.md floods** — lessons are char-bounded (100k). Always dedupe (record_lesson does)
  and keep lessons general/agnostic. Don't write one lesson per trivial action.
- **Not honoring existing lessons** — if a lesson is in MEMORY.md, the conscious LLM must
  act on it. Repeating a known mistake is a failure.

## Verification Checklist

- [ ] `RULE.md` + `docs/MIND_ARCHITECTURE.md` are the authoritative specs — code that
      contradicts them is wrong.
- [ ] No new hardcoded map/coord/item literals in `*.py` (grep the diff before commit).
- [ ] `self_awareness` is populated on `RuntimeState` (lifecycle.py) and injected into LLM
      calls (llm/manager.py + model_router.py).
- [ ] Completed-action hook writes MEMORY.md lessons on failure/refusal (pdca_loop.py).
- [ ] New skills created by agents land under `skills/<domain>/SKILL.md` with valid
      frontmatter (name + description).
- [ ] SOUL.md + MEMORY.md are loaded + prepended on every conscious-tier reasoning call.

## One-Shot Recipes

**Verify SOUL/MEMORY injection works:**
```bash
cd /home/lot399/openkore-ai-v3/AI_sidecar/ai_sidecar
python -c "from ai_sidecar.memory.self_awareness import SelfAwareness; \
from pathlib import Path; sa=SelfAwareness(Path('memory')); \
print('SOUL', len(sa.soul)); print('MEMORY', len(sa.memory_entries), 'entries'); \
print(sa.inject('test prompt')[:200])"
```

**Write a lesson (self-learning write-path test):**
```bash
cd /home/lot399/openkore-ai-v3/AI_sidecar/ai_sidecar
python -c "from ai_sidecar.memory.self_awareness import SelfAwareness; \
from ai_sidecar.autonomy.post_action_review import record_lesson; \
from pathlib import Path; sa=SelfAwareness(Path('memory')); \
print(record_lesson(sa, 'Test lesson: route via discovered portals, never hardcode coords.'))"
```

**Run the skill-system test suite:**
```bash
cd /home/lot399/openkore-ai-v3/AI_sidecar/ai_sidecar
python -m pytest test_skills_system.py -q
```
