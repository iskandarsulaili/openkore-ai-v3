# BIG-PICTURE CONSCIOUS BRAIN — COMPLETENESS CHECKLIST (2026-08-28)

User mandate: "AI bot should able to see the whole big picture from past, present
and future. Short and long term. Short and far view." + "preemptive and not just
simply reactive." + implement/fix/wire/verify EVERYTHING to completeness. Zero
dormant/dead code allowed.

## THE 3 TIME LAYERS + 2 VIEW DISTANCES
- PAST  = episodic/semantic memory (what happened, what worked/failed) — recall
          injected into Conscious (LLM) decisions.
- PRESENT = live charstatus.json (hp/sp/stats/combat/inventory/map/time) — VERIFIED
          already injected into _llm_cold_start_advisory (pdca_loop.py:9654+).
- FUTURE = multi-horizon planning (tactical 30s / short 5m / medium 30m / long 2h)
          + goal decomposer conflict detection — VERIFIED exists (GoalHorizon,
          pdca _run_loop Horizon loop).
- SHORT VIEW = reflex/heuristic (instant, safety floor) — VERIFIED live.
- FAR VIEW  = LLM conscious advisory (cold-start, gear, npc-dialog, help) — VERIFIED
          called every N cycles (pdca_loop.py:2695/2712/2725/2739).

## GAPS FOUND (verified 2026-08-28)
- [ ] G1: LongTermMemory (memory/long_term_memory.py) INITIALIZED but NEVER
      CONSULTED — get_relevant_context() (line 220) is DEAD CODE, no recall into
      any LLM prompt. PAST layer missing.
- [ ] G2: No MEMORY STORE on significant events — nothing records kills/deaths/
      EXP deltas/gear changes into long-term memory. Memory is empty + never fed.
- [ ] G3: Cold-start LLM advisory prompt lacks the past context block (only
      present charstatus + server facts). Preemption impossible without it.
- [ ] G4: Gear/sustain LLM advisory (_llm_gear_advisory) — verify it also gets
      past context (what gear was bought/failed before) + preemptive trigger
      (stock potions BEFORE farming, not when dying).
- [ ] G5: VERIFY preemptive (not just reactive) paths: death-loop prediction
      (hunting_zone_manager risk), danger-score pre-escape, potion stocking
      before low-HP window. Check they RUN (not dead code).

## BATCHES
- [x] B1 (G1+G2+G3): wire memory store (kill/death/EXP deltas) + recall injection
      into cold-start advisory. Unit test + sidecar restart + live verify.
- [x] B2 (G4): gear advisory past context + preemptive sustain trigger.
- [x] B2.5 (NEW 2026-08-28, user directive): BrainRewardLedger — unified
      punish/reward for ALL brains (conscious_llm, heuristic, reflex,
      subconscious_ml, goal_decomposer, memory, strategy). Kills/deaths/HP-
      critical score every brain; ledger context injected into BOTH LLM
      advisories (self-aware, preemptive); GET /v1/conscious/brain-rewards
      observability; JSONL persisted; 7 unit tests green.
- [ ] B3 (G5): verify preemptive paths live (death-loop, danger pre-escape,
      potion pre-stock). Fix any dormant.
- [ ] B4: full test suite (413+), commit, push. Live EXP/kill verification.

## VERIFY (definition of done)
- Bot banks EXP continuously (kills land, weapon equipped, heals fire).
- LLM advisory prompt CONTAINS memory context (grep log for recall block).
- Memory store rows appear on kill/death (log line or DB/file).
- Preemptive: bot stocks potions/leaves danger BEFORE hp critical (not after).
- All tests green. Everything committed + pushed.
