# STALL → SELF-HEAL CHAIN — COMPLETENESS CHECKLIST (2026-08-28)

Mandate: self-HEALING must detect + act on stalling BEFORE self-learning/
self-improving. Zero dormant/dead/unwired. Every stall class detected →
healed → verified.

## Stall classes to cover
- [ ] C1 EMPTY-MAP: bot on a map with 0 monsters (wedge: map owner flapping
      makes mobs vanish) → must CHANGE MAP (heal), not sit
- [ ] C2 NO-PROGRESS: EXP frozen N minutes while in-game (farming stalled)
      → escalate: change map / new farm / restock
- [ ] C3 STUCK-ROUTE: bot can't route (no path) → exploration scout to
      another map (exists: _emit_exploration_scout, 1843)
- [ ] C4 STUCK-IN-TOWN: 3+ town cycles → retreat/change (exists:
      bot_health_monitor 133)
- [ ] C5 DEATH-SPIRAL: repeated deaths → heal chain (exists: reflex)
- [ ] C6 NO-ATTACK: monsters present but bot never attacks → reset attack
      state / re-target (the fixed range/position gates)

## Chain wiring to verify (detection → action → ack → learn)
- [ ] D1 Empty-map signal: where does the sidecar KNOW the map has 0 monsters?
      (actor probe monster list=0 is bridge-side; sidecar must see it)
- [ ] D2 No-progress signal: EXP delta monitor (exists in _remember_significant
      deltas — needs a STALL branch: if in-game + no EXP delta for N min → emit)
- [ ] D3 Heal action: empty-map → change map (exploration/known-farm/LLM best
      guess), not infinite sit
- [ ] D4 Ack/verify: after the heal action, confirm the bot moved + monsters
      present + EXP resumes; if not, escalate (next map)
- [ ] D5 Learn: the stall + the healed state feed memory/reward (self-improve
      AFTER heal succeeds — order: heal first, then learn)

## Verify live
- [ ] V1 Bot on empty map → sidecar emits map-change within N cycles
- [ ] V2 EXP resumes after map change (or next map tried)
- [ ] V3 No false positives (normal farming not flagged as stall)
- [ ] V4 Tests for each stall class (unit + E2E via live state)

## Batch plan
- B1: audit the current stall-detection surface (health_monitor + pdca_loop +
      exploration_scout) — what EXISTS vs what's dormant
- B2: wire empty-map detection (C1) + heal action (D3)
- B3: wire no-progress detection (C2) + escalation
- B4: ack/verify loop (D4) + learn (D5)
- B5: tests + live verify + checklist mark
