# AGNOSTIC COMPLETENESS CHECKLIST — openkore-ai-v3

**Goal:** Replace ALL hardcoded per-server facts (item IDs, map names, coords, talk sequences, prices) with DB-backed / LLM-driven / data-driven resolution. Zero hardcoded per-server rules (RULE.md). Reconcile, never trim.

## Status Legend
- [x] DONE + verified
- [~] IN PROGRESS
- [ ] PENDING

## Batch 1 — Gear / Weapon / Potion / Sell / Job-change / Maps (heuristic_service.py + gear_progression_planner.py + buyable_items.py)

| # | Site | Hardcoded (before) | Fix (after) | Status |
|---|------|--------------------|-------------|--------|
| 1 | gear_progression_planner `_load_upgrade_paths` | ranked event items (id≥20000, cost 10-20z) top — NOT NPC-buyable | NEW `buyable_items.py` parses RAW server shop scripts → buyable set; planner filters to buyable + price floor ≥10z | [x] |
| 2 | `JOB_CHANGE_2_1`/`JOB_CHANGE_TALK` (ro_mechanics.py) | wrong coords (merchant at prontera,120,200; RAW=alberta_in,58,43) + wrong menu seq (swordman=option2 not r1) | DB-backed `JOB_CHANGE_NPCS` coords + `c` open-dialog; LLM dialog responder picks menu option agnostically | [x] |
| 3 | cold-start + 2-1 job-change paths | `JOB_CHANGE_2_1` wrong coords | `JOB_CHANGE_NPCS` (DB) | [x] |
| 4 | `WEAPON_BUY` weapon-by-class (1701/1301/1201/1501) | hardcoded weapon per class | DB-backed gear_progression_planner (stat/zeny ranked, buyable-filtered) | [x] |
| 5 | death-recovery weapon (1201) | hardcoded Knife | DB-backed gear planner | [x] |
| 6 | death-recovery potion name map {501:Red,...} | hardcoded | knowledge DB lookup | [x] |
| 7 | `_get_potion_id` cold-start (501/502/504) | hardcoded level tiers | DB-backed heal-per-zeny + buyable-filtered (Red 501 best value) | [x] |
| 8 | sell-junk Tool Dealer coords (290,221) + `SELLABLE_JUNK` dict | hardcoded | DB-backed sell NPC (knowledge FACT store) + vendor-value<100z junk detection | [x] |
| 9 | `_is_hunting`/`_audit_is_hunting` prefix lists | hardcoded map prefixes | map_spawns membership + not-town (DB-backed) | [x] |
| 10 | `_town_maps` tuple | hardcoded | DB-backed `_HUNT_TOWNS` | [x] |

## Batch 2 — Recovery / Shop (recovery.py + shop.py)

| # | Site | Hardcoded (before) | Fix (after) | Status |
|---|------|--------------------|-------------|--------|
| 11 | recovery.py `_RECOVERY_ITEMS` (501-512 + heal amounts) | hardcoded | DB-backed: name pattern + itemheal script (HP/SP, plain/rand), lowest item_id per name (canonical base item) | [x] |
| 12 | shop.py `_DEFAULT_SHOP_PRICES` | hardcoded | DB-backed (item_db Buy price, 9835 items) | [x] |
| 13 | shop.py `_AUTO_SELL_TYPES` junk detection | hardcoded name list | DB-backed vendor value (Sell < 100z) | [x] |

## Batch 3 — Crafting (alchemy.py / forging.py / cooking.py) — imported but NOT wired into live path

| # | Site | Hardcoded (before) | Fix (after) | Status |
|---|------|--------------------|-------------|--------|
| 14 | alchemy.py `_ALCHEMY_RECIPES` | hardcoded recipes | DB-backed (item_db) — pending | [ ] |
| 15 | forging.py `_FORGE_RECIPES` | hardcoded recipes | DB-backed — pending | [ ] |
| 16 | cooking.py `_COOKING_RECIPES` | hardcoded recipes | DB-backed — pending | [ ] |

## Batch 4 — Map intelligence / PK avoidance / quest executor (secondary, cold-start fallbacks)

| # | Site | Hardcoded (before) | Fix (after) | Status |
|---|------|--------------------|-------------|--------|
| 17 | map_intelligence.py `_load_default_maps` | hardcoded map DB | DB-backed override (server spawn data) — pending | [ ] |
| 18 | pk_avoidance.py `SafeZone` list | hardcoded town coords | DB-backed — pending | [ ] |
| 19 | quest_step_executor.py `_NPC_LOCATIONS` | hardcoded coords | DB-backed — pending | [ ] |

## Verification
- [ ] All modules import clean
- [ ] Bot farms continuously (EXP/zeny proof)
- [ ] No hardcoded per-server literals in decision paths (grep sweep)
- [ ] Benchmark: loop latency, action throughput
- [ ] List all enabled commands for normal user/player on website
