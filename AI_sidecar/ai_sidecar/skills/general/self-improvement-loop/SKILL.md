---
name: self-improvement-loop
description: "The bot's self-learning/healing/improving loop: SOUL + MEMORY injection, lesson write-back, skill curation."
version: 1.0.0
triggers:
  - self_improvement
  - lesson
  - memory_write
  - skill_creation
  - background_review
when_to_use:
  - an action completed and a lesson is worth persisting
  - the conscious LLM decides something is worth remembering
  - a discovery should become a reusable skill
when_not_to_use:
  - recording server-specific facts (use server_solutions DB store instead)
metadata:
  domain: general
  subdomain: self_improvement
  source: self_awareness
  confidence: 0.95
  tags: [memory, soul, self-learning, self-healing]
---
# Self-Improvement Loop

## The Loop
```
Live observation → Conscious LLM reasons (SOUL + MEMORY injected)
  → action emitted → executed → acked
  → record_lesson() writes MEMORY.md on learning-worthy outcomes
  → MEMORY.md re-injected next call → skills_manager creates SKILL.md from discovery
```

## Pieces
- `memory/SOUL.md` — curated identity + doctrine. Injected VERBATIM every call.
- `memory/MEMORY.md` — curated durable lessons, char-bounded (100,000), `\n§\n`-delimited.
  Written by the conscious LLM when it decides something is worth remembering.
- `memory/self_awareness.py` — `inject()` prepends SOUL+MEMORY; `add_lesson()` appends;
  P2P sink pushes/pulls shared lessons. Wired into `llm/manager.py` + `model_router.py`.
- `autonomy/post_action_review.py` — `record_lesson()` writes MEMORY.md lessons;
  `review_action()` / `review_heal_strategy()` create skills from discoveries.
- `skills/` — the sidecar's OWN skill library (manager/loader/usage/curator).
- `autonomy/pdca_loop.py` (~5469) — completed actions → long_term_memory + a MEMORY.md
  lesson on failure/refusal.

## Discipline
- Lessons are general + server-agnostic. Server-specific facts go in `server_solutions`.
- Dedupe lessons (record_lesson does) — never flood the 100k budget.
- Honor existing lessons — never repeat a known mistake.
- Self-heal: when a subsystem degrades, route around it and flag it.
- Self-improve: every post-action review + role-performance outcome feeds the next decision.
