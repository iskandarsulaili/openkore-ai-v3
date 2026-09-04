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
- [ ] Verify snapshot carries inventory (fly wing count) + zeny
- [ ] Death-loop reflex `fly_wing_escape` branch: if no Fly Wing AND no zeny to
      buy one, fall back to `level_up_first` (farm for zeny)
- [ ] If bot has zeny but no Fly Wing, emit `buy 601` then `use 601`
- [ ] Verify the fallback is DB/agent-driven (no hardcoded map literal)
- [ ] Full test suite passes
- [ ] Commit + push
