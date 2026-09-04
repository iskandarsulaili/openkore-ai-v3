# COMPLETENESS — Executable Survival Strategy (2026-09-04)

## Goal
Close the gap where the LLM decides `fly_wing_escape` but the bot has no Fly
Wing (601) and no zeny to buy one → the decision is unexecutable (log:
"You don't have the Teleport skill or a Fly Wing" repeatedly). The executor
must recognize the unexecutable decision and fall back to `level_up_first`
(farm for zeny → buy Fly Wing → job change), so the LLM's decision is ALWAYS
executable.

## Root cause (verified)
- LLM `_llm_survival_advisory` decided `fly_wing_escape` (bypass lethal field).
- Bot inventory: 0 Fly Wings (601), 0 zeny. Cannot execute.
- Death-loop reflex `fly_wing_escape` branch emits `use 601` but bot has none.

## Checklist
- [x] Verify snapshot carries inventory (fly wing count) + zeny — BotStateSnapshot
      has `inventory_items` (item_id/quantity) + `zeny`. VERIFIED.
- [x] Death-loop reflex `fly_wing_escape` branch: if no Fly Wing AND no zeny to
      buy one, fall back to `level_up_first` (farm for zeny). VERIFIED LIVE:
      LLM decided fly_wing_escape → bot had 0x601 + 0 zeny → falls back to
      level_up_first. Committed `4a4b8f9cc`.
- [x] If bot has zeny but no Fly Wing, emit `buy 601` then `use 601` — added
      `dl_buyfw` buy action when zeny>0. Committed `4a4b8f9cc`.
- [x] Verify the fallback is DB/agent-driven (no hardcoded map literal) — uses
      DB-backed farm_map, defers to conscious tier if none learned.
- [x] Full test suite passes (463 passed, 3 pre-existing cold-start failures)
- [x] Commit + push — `4a4b8f9cc`
- [x] LIVE VERIFIED: bot farming (EXP 15766→15812, attacking Little Poring),
      job change deferred (level_up_first)
