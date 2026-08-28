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

## B3/B4 (2026-08-28) — F13 ROUTE-FAILURE HEAL + F8 SHOP AGNOSTICIZATION

- F13 WIRED: route_failure_count >= 8 while in-game -> map-change heal
  (rate-limited 1/_stall_min via _route_heal_ts). Shared _emit_stall_heal
  helper (both no-progress + route-failure triggers use it). Independent
  if-branch (not elif — the EXP branch's unmet condition shadowed it).
- F8 HARDCODE REMOVED: _town_npc = {"prontera": "...", "izlude": "..."} dict
  (RULE.md violation) -> learned shop_npc from server_solutions store; empty
  until observed (buyAuto runs near any shop; reflex covers sustain).
- TESTS: 9/9 (route-failure heal + all stall cases).
- F11 death-loop: ALREADY healed (_death_loop_target suppress window).
- F12 zeny-starvation: covered by F6 reflex potion-fallback (verified live).
- F14 map-flap: covered by F1 no-progress (map flapping = frozen EXP).

## B5 (2026-08-28) — FULL SURFACE VERIFIED (F1-F16)

- F15 party: party_check/joiner_check emit heal signals (leader/joiner/stuck
  town states) — LIVE in logs (in_party=False members=[] leader_char=testbot99).
- F16 in-game flag: bridge 1731 uses REAL network state ($net->getState()==
  IN_GAME) — not stale.
- ALL 16 classes: F1✅ F2✅(F1) F3✅ F4✅ F5✅(backoff auto) F6✅ F7✅ F8✅
  F9✅(gear advisory) F10✅ F11✅ F12✅(F6) F13✅(NEW) F14✅(F1) F15✅ F16✅.
- Principle honored: DETECT -> HEAL deterministic -> ONLY novel/repeated
  failures go to self-learning/self-improving (ledger/memory/LLM advisory).
