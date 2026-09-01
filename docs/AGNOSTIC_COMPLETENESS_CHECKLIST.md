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

**Verdict:** Recipes are UNIVERSAL game facts (same on every RO server — Red Potion = Empty Bottle + Red Herb everywhere), NOT per-server facts. They do NOT violate RULE.md. CraftingDomain is dormant (never instantiated in the live path). Left as-is — no per-server violation to fix.

## Batch 4 — Map intelligence / PK avoidance / quest executor (secondary, cold-start fallbacks)

**Verdict:** Farm map comes from server_solutions DB (learned), NOT map_intelligence. map_intelligence/pk_avoidance/quest_step_executor are cold-start fallbacks only, not in the live decision path. Left as-is — no per-server violation in the live path.

## Task 7 — Website command list (DONE)
- [x] All 12 group-0 (Player) commands verified: changedress, resurrect, ping, autoloot, autolootitem, iteminfo, mobinfo, whodrops, rates, showexp, commands, party
- [x] Each registered in conf/atcommands.yml (grep -c = 1 each)
- [x] Each granted in conf/groups.yml group 0 Commands block
- [x] Live map-server (started 20:02) loaded config (mtime 15:46/15:47) — current
- [x] Website connect page lists all 12 (themes/default/connect/index.php + lang/en_us.php ConnectCommandsList)

## Verification
- [x] All modules import clean
- [x] Bot farms continuously (0 assess() crashes since 20:26 restart, CPU 70% down from 100%+)
- [x] No hardcoded per-server literals in live decision paths (grep sweep)
- [x] Benchmark: loop latency 6-8s (was 75-90s), potion = DB-backed Red 501
- [x] List all enabled commands for normal user/player on website

## Round 2 — Adversarial sweep (2026-09-01, commits ac07026ab..efa1ffcf8)
- [x] get_optimal_weapon fallback → DB gear planner (was hardcoded equipment_progression)
- [x] cold-start step-2 last-resort weapon → cheapest buyable from knowledge DB (was hardcoded 1201)
- [x] equipment-progression upgrade check → DB gear planner
- [x] weapon-latch → resolve item type from knowledge DB by ID (was hardcoded weapon-ID list)
- [x] equipment_progression/loot_values/SELLABLE_JUNK dicts now dead (0 refs)
- [x] pdca_loop.py: 6 town-map tuples + field-prefix routing → DB _is_city_map + real spawn maps
- [x] pro_ro_player_agent job-change fallback town → cities.txt (was hardcoded prontera)
- [x] buyable_items wired at 4 sites, gear planner returns buyable Orcish Sword
- [x] Sidecar restarted 20:26, 0 crashes, bot farming
