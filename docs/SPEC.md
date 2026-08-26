# openkore-ai-v3 — MASTER SPEC & CONTRACT (authoritative, no drift)

> Status: AUTHORITATIVE contract. Updated 2026-08-26. All implementation must
> conform. When in doubt, this doc wins over code comments / stale docs.
> User mandate (2026-08-26): "write all our spec in a doc so we won't drift anymore
> in the future." Reconcile, never trim. Zero mock/stub/placeholder/dormant.

────────────────────────────────────────────────────────────────────────────
## 1. HIGH-LEVEL VISION (unchanged, non-negotiable)

openkore-ai-v3 is a **God-Tier self-adaptive Ragnarok Online bot** that plays
independently with **zero human intervention** from a brand-new level-1 character
through to endgame, **progressing forever** with the game server's mechanics and
growth. It is:
- **Server-agnostic**, **game-agnostic**, **situation-agnostic** — no hardcoded
  server/game/map/coordinate/item/NPC/skill literals in any decision logic.
- **Self-learning**, **self-healing**, **self-improving** — via memory (SOUL.md +
  MEMORY.md + tables + DB) + crowdsource P2P intelligence.
- A **three-tier mind**:
  - **CONSCIOUS (LLM/CrewAI)** = intent-setter + root-cause analyst +
    whole-picture thinker. Decides WHAT/WHY (strategy, goal, farm target, gear,
    job path, cold-start, root-cause). Handles cold-start. Coarse-cadence.
  - **SUBCONSCIOUS (trained ML / RL-DQN)** = ~95% of skilled moment-to-moment
    combat (dodge, kite, skill timing, target selection). Drives action.
  - **REFLEX (hardwired)** = instant/immediate safety floor only (never-die,
    flinch, emergency heal/stand, withdraw). Common, hardcoded, generic.
- **OpenKore core = "the muscle"** — executes commands only. The bridge passes
  commands; it NEVER decides strategy/tactics. OpenKore's own AI must not be the
  source of "what to do."

## 2. AGNOSTIC RULE (HARD, applies to ALL config/decision/knowledge)

**NEVER hardcode server/game/situation-specific facts in *.py or config.**
Config values, lockMap, farm_map, safe_town, potion choice, shop, mob targets,
job path, gear, elements, difficulty — ALL are decided by the LLM/AI agent and/or
**learned from the live server**, persisted to the DB-backed store, and read back
at runtime. They may change depending on problem/solution as decided by the
LLM/AI agent.

The **knowledge base, database and tables GROW with CRUD** — new facts are
created, read, updated, deleted as the bot observes the live server. Nothing is
a fixed literal.

### 2.1 What is allowed to be static (reference knowledge, NOT decisions)
- rAthena **mob_db / item_db / map_index / portal** reference data (the game's
  own tables) loaded from the server's real data files. This is the game's
  mechanical truth, not a decision.
- Generic safety reflex rules (never-die, HP floor, emergency stand) — universal.
- Map-safety ratings / recommended-level ranges as a *baseline* for the LLM to
  weigh, never the final farm decision.

### 2.2 What must be LLM/agent-decided + learned (never a decision literal)
- **lockMap / farm_map / safe_town** (the current blocker — see §5.1)
- potion / consumable choice, buy quantity, shop NPC
- job-change path, gear/equipment plan, stat/skill allocation
- mob target selection, difficulty/element strategy
- routing target between maps

## 3. MEMORY & KNOWLEDGE GROWTH (CRUD everywhere)

### 3.1 Layered memory
- **SOUL.md** — persona/identity/values (injected first). May be updated.
- **MEMORY.md** — self-improvement lessons (deduped, curated). May be updated.
  Written back on fail/refuse/error outcomes via `record_lesson()`.
- **SKILL.md / skills library** — reusable procedure knowledge. May be updated,
  curated, consolidated (skills_curator).
- **Tables + DB** — structured learned facts (server_solutions, exp/drop
  samples, reinforcement stats, crowd-sourced lessons). Grow via CRUD.
- **Crowdsource P2P intelligence** — shared lessons across the fleet.

### 3.2 Injection contract
SOUL.md + MEMORY.md are injected **verbatim into every conscious LLM call**.
Skills are injected by domain relevance. All three files may be updated by the
learning loop; the injected snapshot must reflect the current file contents.

### 3.3 LLM COST TIER (user directive 2026-08-26)
The LLM cost tier is **MAX / UNLIMITED** for the conscious-brain reasoning purpose.
Do NOT throttle/trim conscious-tier LLM reasoning for cost — the cost budget is
not a limiting constraint here. (Applies to the sidecar conscious reasoning path;
the existing per-task max_tokens budgets that guarantee OUTPUT SHAPE are separate
and remain — this directive removes cost-driven throttling, not output-shape caps.)

### 3.3 ServerSolutionsStore (the growth surface)
- `set/get/get_json` CRUD on `server_solutions` table (server_key, slot,
  value_text, value_json, origin, confidence, timestamps).
- **Seed** = initial best-guess from a known-good baseline (origin=seeded).
- **Learn** = overwrite/update from live observation (origin=learned) — the bot
  updates farm_map/safe_town/potion as it discovers the real server.
- **Decision code must read from this store, never hardcode the fallback.**
  The fallback `or "prontera"` / `or "prt_fild08"` is itself a hardcoded literal
  and must be removed — the store must be populated (seeded/learned) before
  decision code needs it.

## 4. ARCHITECTURE / PIPELINE (verified from source)

OpenKore bot → bridge (`plugins/aiSidecarBridge/aiSidecarBridge.pl`) → sidecar
(`ai_sidecar.app:18081`) → PDCA loop (`autonomy/pdca_loop.py`).

- **Bridge** = muscle interface: forwards snapshot, executes commands, emergency
  sit. NEVER decides strategy. RULE.md contract.
- **PDCA loop** = orchestrator: per-bot cycle → conscious advisories (LLM),
  subconscious (RL), reflex (heuristic_service). Queue actions → bridge.
- **heuristic_service** = reflex + cold-start + domain modules.
- **fleet/** = FleetOrchestrator, FleetCoordinator, RoleManager, PartyCoordinator.
- **providers/** = LLM adapters + model_router (CrewAI conscious tier).
- **llm/manager.py** = LLMManager (conscious brain) — MUST read the same config
  as the rest of the sidecar (single source of truth). [FIXED RC1]

## 5. KNOWN DEFECTS / OPEN ITEMS (tracked, being fixed)

### 5.1 [RC3a] lockMap hardcoded → must be agnostic + LLM-decided  [OPEN]
`set lockMap prt_fild05` / `set lockMap {_hunt_map}` literals in
heuristic_service.py (2075, 2402, 2480, 3002, 3020, 5088), bridge default
`aiSidecar_huntingMap 'prt_fild05'` (aiSidecarBridge.pl:943), configs hardcode
lockMap. FIX: lockMap decided by LLM/agent from reachable-farm + server_solutions,
learned + persisted, never a literal. Bridge lockMap stickiness must respect the
sidecar decision.

### 5.2 [RC3] cold-start routing loop  [OPEN]
Bots reach academy door (125,257 warp) but `lockMap prt_fild05` routing pulls
them away before stepping onto the iz_ac01 warp → never register → stuck Lv1/2.
Compounded by 5.1.

### 5.3 Inventory-snapshot gap (false weapon-less)  [OPEN — verify]
Bridge `_build_snapshot_payload` sets `$p{inventory_items}` (names) +
`$p{has_weapon_in_inventory}` at TOP-LEVEL. Sidecar pdca_loop:341 reads
`prog.get("inventory_items")` from the **progression** block (empty). Stored
snapshot confirms: top `inventory_items: []`, `has_weapon: None`. So the sidecar
ALWAYS sees "weapon-less" even when the char owns a knife. Must reconcile the
field location (top-level) with the reader (progression), and confirm the bridge
actually populates inventory (bot log shows items=0 in leakdiag).

### 5.4 cold_start relog loop on live in-game bots  [OPEN]
ColdStartManager.assess may emit relog/char-create for a live in-game bot if the
in-game guard signals (map_known/base_level/in_game/map) aren't reaching it.
Must never relog a live bot.

### 5.5 Dead-brain was RC1 [DONE]
llm/config.py `LLMConfig.from_env()` read only LLM_* env; canonical .env uses
OPENKORE_AI_PROVIDER_*. LLMManager saw no provider → conscious brain dead.
FIXED (95e1230d4): fall back to SidecarSettings for openai+deepseek. Verified
live: LLMManager available=True, _post_json firing.

## 6. VERIFICATION GATES (before declaring anything done)
- LLM conscious advisories fire + enqueue real actions (not just logs).
- Bots make REAL in-game progress: base_level ticks, kills, EXP (DB-confirmed).
- Agnostic audit: no decision literal for farm/lockMap/safe_town/potion.
- Store grows via CRUD; decisions read from store, no hardcoded fallback.
- Fleet orchestration emits directives; leader coordinates.
- Self-improvement loop: lessons written back to MEMORY.md + re-injected.
- Full Python test suite green; bridge perl -c clean.
- All-angles: no new defect introduced by each batch.

## 7. COMMIT/PUSH
Commit after each batch. Push at reasonable stages. Only stage source files,
never runtime artifacts (.env, DBs, logs, .venv, __pycache__, fields/*.dist).

## 8. GAME MECHANICS KNOWLEDGE (rathena-ai-world — READ ONLY, learned facts)
Server: /home/lot399/rathena-AI-world (DO NOT MODIFY). Renewal server.
- New char start_point: iz_int/iz_int01-04 (18,26) academy intro rooms.
- start_items: 1201 (Knife, equipped), 2301 (Adventurer Suit), 23484 (First aid Box 5).
- start_zeny: 0.
- Academy: Cryptura Academy, receptionist iz_ac01 100,39 → Novice_Knife 1243 + 300
  Novice_Potion 569 + gear (TestBotA DB confirms: 1243, 569x300, 2112, 2352, 2414...).
- Bots DO own a knife (1201 or 1243); "weapon-less" is a SNAPSHOT BUG (§5.3), not reality.
- Academy intro room exit: iz_int (51,30) hidden warp → iz_ac01 hall.
- Job change / combat formulas / elements / damage types: read from
  rathena-AI-world db/ (item_db, mob_db, skill_db) + status/formula source (READ ONLY)
  as the agnostic knowledge baseline; NEVER copy server literals into decision code.
