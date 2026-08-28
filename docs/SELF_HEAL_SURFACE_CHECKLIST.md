# SELF-HEAL SURFACE — EVERY CORRECTABLE FAILURE (2026-08-28)

Mandate (user, verbatim): "stalling is NOT the only thing our self-healing
should act on before giving it to self-learning and self-improving" — EVERY
correctable failure must be self-HEALED first; only uncorrectable/learned
patterns go to self-learning/self-improving.

## Failure taxonomy (bot-side, correctable by self-heal)

| # | Failure class        | Detect? | Heal?   | Where                                    |
|---|----------------------|---------|---------|------------------------------------------|
| F1 | No-progress stall     | ✅ NEW  | ✅ NEW  | pdca_loop stall detector (EXP frozen)    |
| F2 | Empty-map (no mobs)   | 🔶 PART | 🔶 PART | (covered by F1 — map-change heal)        |
| F3 | Stuck-route (same map 60cyc) | ✅ | ✅ | combat_monitor reset + scout             |
| F4 | Stuck-in-town         | ✅      | ✅      | bot_health_monitor _is_town_map          |
| F5 | Reconnect backoff     | ✅      | 🔶 PART | timeouts.txt auto (no heal while backoff)|
| F6 | HP-critical / death   | ✅      | ✅      | reflex heal + potion fallback            |
| F7 | No-attack (miss loop) | ✅      | ✅      | attack range/position gates (round 11)   |
| F8 | Inventory full / overweight | 🔶  | 🔶  | (sell/vendor path — verify)              |
| F9 | Weapon broken / no weapon | 🔶 | 🔶  | (gear advisory — verify)                 |
| F10| Job-change eligible   | ✅      | ✅      | job-change chain (round 11)              |
| F11| Death-loop (repeated deaths same map) | 🔶 | 🔶 | (verify — endurance blacklist)           |
| F12| Zeny starvation (can't buy potions) | 🔶 | 🔶 | (verify — economy/reflex)                |
| F13| Stuck on portal (route loop) | 🔶 | 🔶 | (verify — route failure count exists)    |
| F14| Map transition failure (map change loop) | 🔶 | 🔶 | (verify)                                  |
| F15| Party/skill dead state | 🔶 | 🔶 | (verify)                                  |
| F16| disconnected-but-in-game-flag stale | 🔶 | 🔶 | (verify)                                  |

✅ = wired + live. 🔶 = PARTIAL or VERIFY (may fall to learning silently).

## Principle (self-heal BEFORE self-learn)
1. DETECT the correctable failure (event-driven where possible — no server scans).
2. HEAL it deterministically (reflex/heuristic/bridge — no LLM needed for known fixes).
3. ONLY if the heal fails repeatedly / the pattern is novel -> self-learning/
   self-improving (memory, ledger, LLM advisory) with the failure recorded.

## Worklist (this pass)
- [ ] B3: Audit F8-F16 (sell/gear/zeny/portal/map-transition/death-loop) —
      does each DETECT + HEAL, or fall to learning silently?
- [ ] B4: Wire any missing heal (deterministic, event-driven).
- [ ] B5: Test each heal path (unit + live).
- [ ] B6: Verify ledger/memory only get the failures learning SHOULD see.
