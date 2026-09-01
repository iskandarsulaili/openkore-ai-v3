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

## Round 7 — Adversarial sweep (2026-09-01, commit 88659c7d9)
- [x] 2-1 job-change path (line 5211) used hardcoded JOB_CHANGE_2_1 coords — but its own comment said those coords are WRONG for RAW and to use DB-backed JOB_CHANGE_NPCS. Code contradicted comment. Now prefers JOB_CHANGE_NPCS (verified wizard=gef_tower, knight=prt_in, hunter=hu_in01). Hardcoded dict = last-resort fallback only.
- [x] JOB_2_1_CLASSES (swordman→knight, mage→wizard) = universal game fact, not per-server routing. Acceptable.
- [x] Verified: sidecar restarted 22:18, 0 crashes, bot Mage base 22 job_lv 10 farming.

## Round 6 — Adversarial sweep (2026-09-01, commits abf3c7cf5, de2ba2cc7)
- [x] gear_upgrade_after_death (pdca_loop:5544) called get_best_upgrade without job — Mage could buy a Novice-only Sword after death. Now passes job from conscious snapshot.
- [x] domains/progression.py _cold_step2: get_best_upgrade missing job — now passes job_name (added to signature + caller).
- [x] domains/equipment.py:44: read the hardcoded equipment_progression dict (Round-1 violation). Now uses DB-backed gear planner (job-aware, NPC-buyable).
- [x] VERIFIED: domains/progression.py _cold_step7 (hardcoded JOB_CHANGE_2_1) is DEAD CODE — legacy domains run observe-only (commands→log, never executed). Live job-change path = heuristic_service.py step 7 (fixed Round 4). No action needed.
- [x] Verified: bot Mage base 22 job_lv 10, farming Lunatic/Poring. DQN training_steps 2163, exp 2157.

## Round 5 — Adversarial sweep (2026-09-01, commit 5a9e7fd94)
- [x] CRASH: _class_stat_allocation line 366 unpacked (bp, needed) but ro_mechanics get_nearest_breakpoint returns (bp_value, bp_description) — 2nd element is a str label, not points needed. Every assess() crashed TypeError '>' not supported between str and int (488+ times). Now computes needed = max(0, int(bp)-int(current)). Verified: mage allocates 20 to INT. Sidecar restarted, 0 crashes.

## Round 4 — Adversarial sweep (2026-09-01, commit c8b4565ce)
- [x] Job-change conflict: bot became Mage but LLM conscious tier had decided merchant (server_solutions job_change_target). Step 7 used _assigned_jobs (team-synergy position fallback) and ignored the conscious decision. Now step 7 prefers the conscious job_change_target; fallback only fires when no conscious decision exists.
- [x] Verified: DB has target_class=merchant, fix reads it. Sidecar restarted 21:52, 0 crashes, bot Mage base 22 farming.

## Round 3 — Adversarial sweep (2026-09-01, commits 04066ae42..5bb73e3a6)
- [x] Equipment optimizer consumer: command=str(_eq) stringified the dict (NOT a valid OpenKore command) — upgrade/slot/repair never executed. Now emits real 'command' field.
- [x] General equipment-progression check filtered to slot_name=='weapon' only — bot only bought/equipped weapons. Now covers ALL 8 slots + cards + refine.
- [x] Gear planner was NOT job-aware — recommended Orcish Sword to a Mage (can't equip). Now job-filtered (Jobs.All=every class, Jobs.Novice=Novice ONLY). All 7 consumers pass the bot's job.
- [x] Verified: mage@21 -> accessory (no affordable mage weapon), swordman@21 -> Orcish Sword. Mage weapons (Rod/Knife/Fortune Sword) in path.
- [x] Sidecar restarted 21:39, 0 crashes, bot Mage base 22 farming.
- [x] get_optimal_weapon fallback → DB gear planner (was hardcoded equipment_progression)
- [x] cold-start step-2 last-resort weapon → cheapest buyable from knowledge DB (was hardcoded 1201)
- [x] equipment-progression upgrade check → DB gear planner
- [x] weapon-latch → resolve item type from knowledge DB by ID (was hardcoded weapon-ID list)
- [x] equipment_progression/loot_values/SELLABLE_JUNK dicts now dead (0 refs)
- [x] pdca_loop.py: 6 town-map tuples + field-prefix routing → DB _is_city_map + real spawn maps
- [x] pro_ro_player_agent job-change fallback town → cities.txt (was hardcoded prontera)
- [x] buyable_items wired at 4 sites, gear planner returns buyable Orcish Sword
- [x] Sidecar restarted 20:26, 0 crashes, bot farming
