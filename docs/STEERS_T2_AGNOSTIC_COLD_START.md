# STEERS_T2 — 3-TIER COLD-START: CONSCIOUS LLM OWNS AGNOSTIC DECISION (2026-08-25)

## Directive (founder, repeated)

> "The data driven are from the tables and database, but LLM are the one who should
> act like conscious mind to solve the cold-start issue on different server or
> situation. They must be agnostic, not hardcoded."
>
> "Remember, we have different layer. Conscious (LLM), subconscious (ML model learn
> from success/fail reward/punish from AI bot behavior or LLM) and reflex
> (rule-based that need instant action)."

## What this means (founder-corrected 3-tier split)

- **Conscious (LLM)** = the DECISION-MAKER, driven by **specialized AI AGENTS inside
  openkore-ai-v3** (CrewAI agents: pro_ro_player, economy, combat, mission, world —
  NOT Hermes, NOT generic). These agents are **issue/solution agnostic**: they reason
  from FACTS (level, inventory, map, portals.txt/DB, server name) to produce the plan
  on ANY server — they are NOT hardcoded rules for a specific issue/map. Examples:
  cold-start advisory, gear/sustain advisory, team-play, farm choice.
- **Subconscious (ML/RL)** = learns from **success/fail reward/punish** of the bot's
  own actions (or the LLM's decisions) — the trained "what usually works" layer
  (DQN/reinforcement learning, promotion pipeline). Facts alone; no strategy.
- **Reflex (hardcoded rule-based)** = INSTANT-action safety floor ONLY (never-die,
  sit on low HP, portal_walk_lock during an active warp walk). Immediate, never
  strategic, never decides the plan.

A rule that bakes a server-specific decision (e.g. "izlude academy door 125 257",
"farm prt_fild08", "buy item 501") is a VIOLATION — it must live as a FACT in the
knowledge store / server_solutions DB and be consumed by the LLM agents (conscious)
or a generic reflex, never hardcoded as the decision.

## Current-state gap (verified 2026-08-25, file:line)

- `AI_sidecar/ai_sidecar/domains/progression/cold_start.py:295 ColdStartManager.assess()` is
  a PURELY DETERMINISTIC character-creation state machine — NO LLM call. It is labeled
  "conscious" but is actually hardcoded rules.
- `AI_sidecar/ai_sidecar/autonomy/heuristic_service.py` has HARDCODED cold-start branches:
  - line ~2894 `_cs_stable_key` (account-keyed, collides cross-char)
  - line ~2947→2979 academy-door resolution + `move 125 257` (map-specific izlude)
  - line ~2062 iz_ac01_a academy-room, ~2490 academy-on-map attack enable
  - line ~2342 goal-decomposer `move prt_fild08` (competing move emitter)
  These are the DRIFT the directive forbids — server-specific facts baked as rules.

## Target architecture (to implement)

1. **LLM conscious decision**: add an `llm_consult_startup` hook — when a bot is in
   cold-start (no weapon / level<baseline), the LLM (combo/deepseek-v4-flash, budget) is
   asked ONCE (cache per bot+state): given {level, inventory, map, portals-from-this-map,
   server}, produce an agnostic plan ("get starter kit", "farm 50z on reachable field",
   "talk to X"). Data-driven facts injected; NO hardcoded map/NPC literals in the rule.
2. **Deterministic helpers become FACTS**, not decisions: `_cold_start_academy_door`
   (tables/portals.txt) is a fact-source the LLM plan can reference, not the decider.
3. **Reflex stays**: portal_walk_lock (bridge aiSidecarBridge.pl), never-die, sit — instant
   only, never strategic.
4. **Subconscious**: existing DQN/reinforcement learning layers consume cold-start
   OUTCOMES (did the plan reach the farm / level faster?) as reward signal — wire the
   cold-start plan-result into the reward pipeline.
5. **Docs drift-guard**: this file + RULE.md + FOUNDER_STEERS.md must state the 3-tier
   split so no future contributor re-hardcodes a server fact as a decision rule.

## Migration order (avoid breakage)

- [ ] 1. Document (this file) — DONE
- [ ] 2. Add the LLM cold-start consult (cached, once-per-bot-per-state) with a kill-switch
      `cold_start.llm_priority` (default on) so it can be disabled if LLM flaky.
- [ ] 3. Reframe deterministic academy/hunt helpers as FACT sources consumed by the plan.
- [ ] 4. Wire cold-start plan outcome into the subconscious reward pipeline.
- [ ] 5. Sweep + remove remaining hardcoded server-specific literals in cold-start branches.
- [ ] 6. E2E on izlude + a fresh unknown map to prove agnostic.