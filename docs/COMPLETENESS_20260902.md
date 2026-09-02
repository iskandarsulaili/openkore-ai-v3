# COMPLETENESS CHECKLIST — openkore-ai-v3 map-login + stat-allocation 2026-09-02

**Goal:** Fix the map-login reconnect loop (bot stuck, 0x006A reject), correct job-aware
stat allocation (Mage must get INT not DEX), verify 2-1 job change completes, prove live.

## Round A — Map-login reconnect loop (0x006A)
- [x] DIAGNOSED: bot in reconnect loop — map-server reject 0x006A (error 1/2, alternates)
- [x] ROOT CAUSE: stale learned 23-byte layout invalidated by blind rotation (19→23→26→19);
      the auto-adapt cleared a confidence-1.0 learned layout instead of trusting it
- [x] VERIFIED: correct 23-byte 0x0436 (id@0 account@2 char@6 sess@10 [0]@14 tick@18 sex@22) lands
- [x] PROVEN: 'TestBotA' logged in (map-server 08:44:40), got Novice Poring Card, stable (1 reconnect)

## Round B — 2-1 job change (Mage job_lv 10 -> Wizard)
- [ ] Trigger fires (first-class at job_lv>=10 routes to 2-1 class)
- [ ] NPC coords resolved DB-backed (gef_tower)
- [ ] LLM dialog responder completes the menu
- [ ] Verified: job becomes Wizard

## Round C — Stat allocation job-awareness (Mage = INT, not DEX)
- [ ] Trace source of `stat_add dex` on Mage
- [ ] Verify _class_stat_allocation returns INT for mage
- [ ] Eliminate hardcoded DEX fallback if present
- [ ] Verified live: Mage allocates INT

## Round D — Cleanup
- [ ] Remove debugMapLogin diagnostic + flag
- [ ] Update checklist + commit
