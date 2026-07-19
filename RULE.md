# RULE.md — Developer Reference for openkore-ai-v3

## Architecture: Three-Layer Separation

```
┌──────────────────────────────────────────────────────────┐
│                    AI SIDECAR (Python)                    │
│  PDCA Loop · CrewAI Agents · Goal Stack · Decision Svc   │
│  Handles ALL decisions: config, routing, job change, etc │
└──────────────────────────────────────────────────────────┘
         ↑ actions (action_queue)        ↑ snapshot data
┌──────────────────────────────────────────────────────────┐
│              BRIDGE PLUGIN (Perl) ONLY FOR:               │
│  1. Monitor/report state to sidecar (snapshots)          │
│  2. Instant reflexes only (sub-second emergency)         │
│     e.g. survival flee when HP < 15% + aggro > 3         │
└──────────────────────────────────────────────────────────┘
         ↑ commands                    ↑ state
┌──────────────────────────────────────────────────────────┐
│                    OPENCORE BOT (Perl)                    │
│  Movement · Combat · Loot · NPC interaction              │
└──────────────────────────────────────────────────────────┘
```

## Hard Rules

### 1. Bridge Plugin is LIMITED
The bridge (`plugins/aiSidecarBridge/aiSidecarBridge.pl`) ONLY does:
- **Monitor/report**: Send snapshot data to sidecar (vitals, inventory, position, progression, skills, actors)
- **Instant reflexes**: Actions that need sub-second response (emergency survival flee when HP < 15% + aggro > 3, force-stand on field maps)
- **Execute commands**: Receive and execute `set <key> <value>` config changes from sidecar
- **NEVER**: Hardcoded config changes, routing decisions, selling logic, job change logic — those belong in sidecar

### 2. AI Sidecar Handles EVERYTHING Else
The sidecar (`AI_sidecar/`) handles ALL decisions:
- **Config changes**: Detect overweight → queue `set sellAuto 1` action → bridge hot-reloads
- **Routing**: Detect stuck in town → queue `move <hunting_map>` action
- **Job change**: Detect job_level ≥ 10 for Novice → route to job change NPC via database
- **Healing**: Detect low HP → queue potion use or sit
- **Skill allocation**: Detect unspent skill/stat points → allocate
- **Economy**: Detect full inventory → queue sell/restock actions

### 3. Zero Hardcoded Values
- **NPC positions**: Use `tables/job_change_locations.txt` — never hardcode coordinates
- **Map names**: Use `zone_ladder` or `GameKnowledgeService` — never hardcode `"prt_fild08"`
- **Item/potion names**: Use dynamic `GameKnowledgeService` — never hardcode `"Red Potion"`
- **Skill names**: Use skill database — never hardcode skill IDs
- **Level requirements**: Use `ro_knowledge.assess_job_advancement()` — never hardcode `"level >= 10"`
- **Credentials**: Single `.env` source — never duplicate in config files

### 4. 100% Data Flow Completeness
Every field in every Pydantic model MUST be populated:
- **Vitals**: `hp`, `hp_max`, `hp_ratio`, `sp`, `sp_max`, `sp_ratio`, `weight`, `weight_max`, `weight_ratio`, `level`, `base_level`, `job_level`, `zeny`
- **Position**: `map`, `x`, `y`
- **Combat**: `ai_sequence`, `target_id`, `is_in_combat`, `aggro_count`
- **Inventory**: `zeny`, `item_count`, `weight`, `weight_max`, `weight_ratio`, `overweight_ratio`
- **Progression**: `job_id`, `base_level`, `job_level`, `base_exp`, `base_exp_max`, `job_exp`, `job_exp_max`, `skill_points`, `stat_points`, `job_name`
- **Skills**: name + level for each known skill
- **Actors**: nearby entities with type, name, hp, distance

If a field is `None`/`0` when it shouldn't be, trace the gap in: bridge → Pydantic model → conscious engine → normalizer → PDCA/planner.

### 5. Agent Synergy — No Conflicts

| Agent | Role | Reads | Writes | Must not conflict with |
|-------|------|-------|--------|----------------------|
| **Survival reflex (bridge)** | Emergency flee when dying | HP, aggro_count, map | lockMap, AI mode toggle | Pro RO lockMap (has 180s grace window) |
| **Pro RO Player (PDCA)** | Recommends hunting maps | Level, job, location | lockMap, move commands | Survival reflex (grace period protects this) |
| **Progression Planner (CrewAI)** | Job change, equipment | `job_change_available` signal | move commands to NPC | Survival (should not flee from job NPC) |
| **Goal Stack** | Priority-ordered goal selection | Assessment results | GoalDirective entries | Must not create redundant goals |
| **Cold Start (PDCA)** | Initial routing on connect | Latest snapshot | lockMap, move | Pro RO high-confidence (cold start backs off after confident action) |

**Goal priority order** (deterministic):
1. **Survival** — HP < 35% or dead/disconnected
2. **Job Advancement** — job level >= requirement and skill/stat points pending
3. **Opportunistic Upgrades** — economy/gear upgrade opportunities
4. **Leveling** — hunting progression (default when nothing urgent)

### 6. Reward/Punish System (Self-Supervised)
- **objective_max_age_cycles**: Goals expire after N PDCA cycles (punishment for stale/unachievable goals)
- **priority_decay_per_cycle**: Goal priority decays each cycle (reward cycles down if blocker unresolved)
- **reflex_actions_suppressed**: Bridge suppresses duplicate reflex actions — counts as negative feedback
- **HighConf counter**: Actions with confidence ≥ 0.8 signal the cold start to back off — success signal
- **ProvFail counter**: LLM provider failures degrade decision quality — must be monitored

## Coding Standards

### Perl (Bridge Plugin)
- Always `use strict; use warnings;` at file scope
- Declare all variables with `my` before first use in every sub
- `quotemeta()` is a FUNCTION — use `\Q...\E` inside regex, never `quotemeta()` as literal text
- Check both `$char->{key}` and `$::config{key}` for initialization before arithmetic
- Log at level 1 for production (`debug "msg", 'aiSidecarBridge', 1`)
- Paths: use `$::Settings->{tablesPath}` with fallback `|| 'tables'`

### Python (Sidecar)
- All Pydantic models should have complete field coverage — `extra="ignore"` is the default but don't rely on it to swallow missing data
- Use `getattr(obj, field, default)` for safe attribute access on snapshot objects
- When enriching signals in `crew_manager.py`, handle both dict and BotStateSnapshot objects
- Use `Path` for file paths, never hardcoded strings
- Imports at module top, not inline (exception: `pdca_loop.py` dynamic routing functions may inline)

## Git Workflow
- One semantic change per commit (not "fix stuff")
- Commit messages: `fix:` for bugs, `feat:` for features, `fix:` for data flow fixes
- Include the WHAT and WHY in the commit body, not just the title
- Revert bridge-only-appropriate changes that cross the architecture boundary

## Debugging Checklist
When a bot isn't behaving as expected, check in this order:
1. **Bridge snapshot** — is the data reaching the sidecar? Check `sidecar.log` for snapshot count
2. **Pydantic model** — is the field defined? Check `contracts/state.py`
3. **Conscious engine** — is the signal being extracted? Check `conscious_engine.py`
4. **Crew manager** — is the signal being enriched? Check `crew_manager.py`
5. **Goal stack** — is the correct goal being selected? Check `goal_stack.py`
6. **PDCA loop** — is the action being generated? Check `pdca_loop.py` action queue
7. **Bridge execution** — is the bridge receiving and executing the action?

## Key Files

| File | Purpose |
|------|---------|
| `plugins/aiSidecarBridge/aiSidecarBridge.pl` | Bridge — monitor/report + reflexes only |
| `AI_sidecar/ai_sidecar/app.py` | FastAPI entrypoint |
| `AI_sidecar/ai_sidecar/contracts/state.py` | Pydantic models (Vitals, InventoryDigest, etc.) |
| `AI_sidecar/ai_sidecar/conscious_engine.py` | Bot state awareness |
| `AI_sidecar/ai_sidecar/crewai/crew_manager.py` | Signal enrichment for CrewAI agents |
| `AI_sidecar/ai_sidecar/crewai/agents/progression_planner_agent.py` | Job change + equipment agent |
| `AI_sidecar/ai_sidecar/autonomy/pdca_loop.py` | PDCA cycle — action generation |
| `AI_sidecar/ai_sidecar/autonomy/goal_stack.py` | Goal prioritization |
| `AI_sidecar/ai_sidecar/autonomy/ro_knowledge.py` | RO game knowledge (job change, leveling) |
| `AI_sidecar/ai_sidecar/autonomy/decision_service.py` | Decision service orchestration |
| `tables/job_change_locations.txt` | Job change NPC coordinates (ALL classes) |
| `knowledge/rathena_db/db/` | rAthena game data (mobs, items, skills, jobs) |
| `.bot_profiles/*/control/config.txt` | Per-bot OpenKore config |
| `logs/` | Bot logs, sidecar log, checkpoints |

## Anti-Patterns to Avoid
- ❌ Adding business logic to the bridge (selling, routing, job change)
- ❌ Hardcoding any value that has a database/tables file
- ❌ Editing config.txt files directly — the sidecar AI should do it via `set` commands
- ❌ Leaving Pydantic model fields unpopulated — 100% completeness
- ❌ Adding new agents without checking for goal conflicts with existing agents
- ❌ Building a fix without checking if the data flows end-to-end first
