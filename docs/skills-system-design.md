# AI Sidecar Skill System — Design & Implementation Plan

> Based on deep analysis of Hermes Agent's skill system (87 skills, 8 skill-related tools, 2016-line curator, 991-line background review fork) and OpenKore AI v3's sidecar architecture (108 Python modules, 25 API routers, 12 CrewAI agents, 83 bridge functions).

---

## 1. Hermes Skill System Architecture (Reference)

### 1.1 Core Components

| Component | File | Lines | Function |
|-----------|------|-------|----------|
| `skill_manager_tool.py` | Agent-facing create/edit/patch/delete | 59,651 | SKILL.md CRUD with frontmatter validation, security scan, atomic writes |
| `skills_tool.py` | List/view skills (progressive disclosure) | 66,889 | `skills_list` returns metadata only; `skill_view` loads full content on demand |
| `skill_usage.py` | Usage telemetry + lifecycle states | 35,737 | `.usage.json` tracks use_count, view_count, patch_count, state (active/stale/archived) |
| `skill_provenance.py` | Write-origin tracking | 2,602 | ContextVar distinguishes foreground (user) vs background_review (agent) writes |
| `agent/curator.py` | Background skill maintenance | 2,016 | Auto-transitions: active→stale→archived; LLM consolidation; pre-run tar.gz backup |
| `agent/background_review.py` | Post-turn self-improvement fork | 991 | After each turn, forks agent to evaluate: "should any skill/memory be saved?" |
| `skills_guard.py` | Security scanner for downloaded skills | 46,724 | Regex-based static analysis; trust tiers: builtin/trusted/community |
| `skills_hub.py` | Registry integration | 162,876 | Install/publish/browse skills from remote registries |

### 1.2 Skill Lifecycle

```
[Agent creates skill] ──→ ACTIVE (use_count=0)
                              │
                              ▼ use_count increments on every skill_view/skill_manage
                         ACTIVE (use_count=N)
                              │
                              ▼ unused > stale_after_days (config, default ~14)
                         STALE (removed from context assembly)
                              │
                              ▼ unused > archive_after_days (config, default ~30)
                         ARCHIVED (moved to .archive/ tar.gz)
                              │
                              ▼ (optional) curator LLM review
                         DELETED or CONSOLIDATED into broader skill
```

### 1.3 Provenance & Write Guards

- **Foreground** (`_write_origin = "foreground"`): Normal agent tool calls. Skills belong to user. Curator never touches.
- **Background review** (`_write_origin = "background_review"`): Forked agent evaluating conversation. Skills are agent-created. Curator manages lifecycle.
- **Guards**: Background review must `read` a file before `write`; can't create skills purely from inference. Pinned skills prevent auto-transitions but permit edits.

### 1.4 Context Loading (Progressive Disclosure)

1. `skills_list(category=None)` — returns `[{name, description, metadata.tags}]` (tier 1)
2. Agent picks skill by name → `skill_view(name)` — loads full SKILL.md + linked files (tier 2-3)
3. Only skills whose `triggers` match the current task are auto-loaded by context assembler

### 1.5 Curator Config

```yaml
curator:
  enabled: true
  interval_hours: 168  # weekly
  min_idle_hours: 1
  stale_after_days: 14
  archive_after_days: 30
  consolidate: false    # LLM review pass (opt-in)
  backup:
    enabled: true
    path: ~/.hermes/skills/.curator_backups/
```

---

## 2. OpenKore AI v3 System Architecture (Target)

### 2.1 Component Map

```
┌────────────────────────────────────────────────────────────────┐
│  Bridge Plugin (Perl, 83 subroutines)                          │
│  plugins/aiSidecarBridge/aiSidecarBridge.pl                    │
│                                                                │
│  REFLEXES (direct in plugin):                                  │
│  • HP < 20% → POST /v1/discover/heal                          │
│  • Party invite from console → auto-accept                     │
│  • Sibling bot identification                                  │
│                                                                │
│  IMMEDIATE ACTIONS:                                            │
│  • Stand up / enable auto mode                                 │
│  • Config apply (attackAuto, itemsTakeAuto, etc.)              │
│  • Execute sidecar-returned command                            │
│                                                                │
│  DATA SOURCE:                                                  │
│  • Reads tables/ (portals.txt, npcs.txt, shops, monsters...)   │
│  • Pushes table data to sidecar via /v1/discover/tables/ingest │
└──────────────────────────┬─────────────────────────────────────┘
                           │ HTTP (events + actions)
                           ▼
┌────────────────────────────────────────────────────────────────┐
│  AI Sidecar (FastAPI, Python, 108 modules)                     │
│  AI_sidecar/ai_sidecar/                                        │
│                                                                │
│  API ROUTERS (25 files):                                       │
│  • discovery.py — Heal strategy, tables ingest                 │
│  • actions.py — Action queue                                   │
│  • crewai_v2.py — CrewAI agent integration                     │
│  • autonomy.py — PDCA loop control                             │
│  • npc_dialog.py — NPC interaction                             │
│  • party.py — Party management                                 │
│  • combat.py — Combat decisions                                │
│  • planners_v2.py — Pro RO LLM planning                        │
│  • ... and 17 more                                             │
│                                                                │
│  CREWAI AGENTS (12 agents):                                    │
│  • pro_ro_player_agent.py — 20-year RO veteran                 │
│  • strategic_planner_agent.py — Long-term strategy             │
│  • tactical_commander_agent.py — Combat tactics                │
│  • economy_agent.py — Market decisions                         │
│  • navigation_agent.py — Map routing                           │
│  • safety_agent.py — Risk detection                            │
│  • progression_planner_agent.py — Build planning               │
│  • resource_manager_agent.py — Inventory management            │
│  • questing_agent.py — Quest automation                        │
│  • social_agent.py — Player interaction                        │
│  • combat_agent.py — Combat execution                          │
│  • macro_engineer_agent.py — Macro generation                  │
│  • fleet_liaison_agent.py — Multi-bot coordination             │
│  • opportunity_trader_agent.py — Vendor arbitrage              │
│  • state_assessor_agent.py — State evaluation                  │
│  • command_emitter_agent.py — Command dispatch                 │
│  • social_coordinator_agent.py — Multi-bot social              │
│  • manager_agent.py — Supervisor agent                         │
│                                                                │
│  PLANNER MODULES:                                              │
│  • context_assembler.py — LLM context builder                  │
│  • plan_generator.py — Plan generation                         │
│  • self_critic.py — Self-critique                              │
│  • reflection_writer.py — Post-action reflection               │
│                                                                │
│  AUTONOMY LOOP:                                                │
│  • pdca_loop.py — Plan-Do-Check-Act cycle                     │
│  • plan_executor.py — Execute plans                            │
│  • goal_stack.py — Goal prioritization                         │
│  • progress_tracker.py — Progress monitoring                   │
│  • ro_knowledge.py — 13 Pro RO axioms                          │
│                                                                │
│  ECONOMY:                                                      │
│  • economic_engine.py — Market simulation                      │
│  • npc_shop_db.py — Shop database                              │
│  • vending_automation.py — Automated vending                   │
│  • market_arbitrage.py — Price arbitrage                       │
│  • farming_selector.py — Best farming targets                  │
│                                                                │
│  COMBAT:                                                       │
│  • elemental_matrix.py — Element advantages                    │
│  • monster_db.py — Monster database                            │
│  • card_db.py — Card effects                                   │
│  • mvp_mechanics.py — MVP boss fights                          │
│  • skill_rotation.py — Skill rotation optimization             │
│  • threat_targeting.py — Target selection                      │
│                                                                │
│  MEMORY:                                                       │
│  • long_term_memory.py — SQLite-backed memory                  │
│  • semantic_store.py — Semantic embeddings                    │
│  • episodic_store.py — Episode recall                          │
│  • retrieval.py — Memory retrieval                             │
│                                                                │
│  FLEET:                                                        │
│  • coordinator.py — Multi-client coordination                  │
│  • swarm_ai.py — Swarm behaviors                               │
│  • party_coordinator.py — Party automation                     │
│  • outcome_reporter.py — Result reporting                      │
└────────────────────────────────────────────────────────────────┘
```

### 2.2 Current Knowledge Flow

```
OpenKore tables/ ──→ Bridge reads ──→ HTTP POST ──→ Sidecar (in-memory cache)
                                                        │
                                  ro_knowledge.py ◄─────┤ (hardcoded axioms)
                                                        │
                                  context_assembler.py ◄─┤ (loads into LLM prompt)
                                                        │
                                  discovery.py ◄────────┤ (heal strategy endpoint)
```

**Problem**: Knowledge is hardcoded in `ro_knowledge.py` (13 axioms) and `pro_ro_player_agent.py` (elemental matrices, size penalties). When a CrewAI agent discovers server-specific information (e.g., "this server's Healer is at 159,193"), it can only update hardcoded Python files. No persistence between sessions.

### 2.3 Existing Integration Points for Skills

| Point | File | What it does |
|-------|------|-------------|
| Context assembly | `context_assembler.py` line 165 | `context["skills"] = operational.get("skills", {})` — already has a skills slot! |
| RO knowledge load | `ro_knowledge.py` | `ROKnowledgeBundle` loads from `autonomy/data/*.json` |
| Heal resource | `heal_resource_loader.py` | Reads `knowledge/knowledge.json` for healing items |
| Discovery tables | `discovery.py` | In-memory `_server_tables` cache from bridge pushes |
| CrewAI agent dispatch | `crew_manager.py` | Routes tasks to best-suited agent profile |
| Post-action reflection | `planner/reflection_writer.py` | Writes reflection after plan execution |
| Learning DB | `learning/shared_learning_db.py` | SQLite DB for cross-bot shared learning |
| Experience DB | `experience_db.py` | Server-specific experience tracking |

---

## 3. Proposed Skill System for AI Sidecar

### 3.1 Directory Layout

```
AI_sidecar/ai_sidecar/
├── skills/                          # Skill store (created at startup)
│   ├── healing/                     # Domain category
│   │   └── server-heal-strategy/    # Individual skill
│   │       ├── SKILL.md             # Instructions (YAML frontmatter)
│   │       └── references/
│   │           └── npc_data.txt     # Supporting data
│   ├── grinding/
│   │   └── hunting-zone-selection/
│   │       └── SKILL.md
│   ├── navigation/
│   │   └── portal-discovery/
│   │       └── SKILL.md
│   ├── economy/
│   │   └── vendor-arbitrage/
│   │       └── SKILL.md
│   ├── .usage.json                  # Metrics tracking
│   └── .archive/                    # Stale skills
│
├── skills_manager.py               # create/read/update/delete/patch skills
├── skills_usage.py                 # .usage.json tracking (use_count, view_count, patch_count)
├── skills_loader.py                # Load matching skills into context_assembler
├── skills_curator.py               # Background stale detection + lifecycle transitions
├── background_review.py            # Post-action background review (forks to evaluate skill changes)
│
├── api/routers/
│   └── skills.py                   # HTTP API: POST /v1/skills/manage, GET /v1/skills/list, GET /v1/skills/view
│
├── planner/
│   └── context_assembler.py        [MODIFY] Load relevant skills into LLM context
│
└── autonomy/
    ├── post_action_review.py       [MODIFY] Trigger background review after key actions
    └── pdca_loop.py                [MODIFY] Call skill curator periodically
```

### 3.2 SKILL.md Format

```yaml
---
name: server-heal-strategy
description: "Optimal healing strategy for this RO server based on discovered NPCs and shop data"
version: 1.0.0
author: crewai_discovery_agent
created_at: "2026-07-17T17:30:00Z"
triggers:
  - "heal_strategy_requested"
  - "discovery_npc_found"
  - "low_hp"
when_to_use:
  - hp_ratio < 0.30
  - server is unknown or has no existing heal skill
when_not_to_use:
  - bot has potions and is in a dungeon (use potion, not NPC)
metadata:
  domain: healing
  subdomain: npc_interaction
  confidence: 0.85
  discovery_count: 3
  peer_confirmations: 2
  server: "origin-ro"
  tables_source: "npcs.txt, npc_shops.txt"
  tags: [healing, npc, prontera, beginner]
  related_skills: [economy-potion-purchase, grinding-poring]
---
```

### 3.3 .usage.json Schema

```json
{
  "server-heal-strategy": {
    "state": "active",
    "created_at": "2026-07-17T17:30:00Z",
    "last_activity_at": "2026-07-17T17:45:00Z",
    "use_count": 47,
    "view_count": 12,
    "patch_count": 3,
    "confidence": 0.85,
    "provenance": "foreground",
    "pinned": false
  },
  "grinding-poring": {
    "state": "stale",
    "created_at": "2026-07-10T10:00:00Z",
    "last_activity_at": "2026-07-14T10:00:00Z",
    "use_count": 12,
    "view_count": 3,
    "patch_count": 0,
    "confidence": 0.6,
    "provenance": "background_review",
    "pinned": false
  }
}
```

### 3.4 Lifecycle States

```
PROVENANCE: foreground (user/manual) | background_review (agent/auto) | bundled (ships with system)
                                      
           CREATION BY AGENT          
  crewai_discovery_agent finds NPC ──→ skills_manager.create()
  crewai_navigation_agent maps route ──→ skills_manager.create()
  crewai_economy_agent finds price ──→ skills_manager.create()
              │
              ▼
          ACTIVE (default)           
              │                      
              ▼ use every cycle (use_count increments)
          ACTIVE (gaining confidence)
              │                      
              ▼ unused > stale_after_days (config: 7 days)
          STALE (removed from context)
              │                      
              ▼ unused > archive_after_days (config: 14 days)
          ARCHIVED → .archive/ dir   
              │                      
              ▼ (background LLM review) 
          DELETED or CONSOLIDATED    
```

### 3.5 Integration Flow

```
┌─────────────────────────────────────────────────────────────────┐
│  POST-ACTION REVIEW (background_review.py)                       │
│                                                                  │
│  After every PDCA cycle completion:                              │
│  1. Take action snapshot (what happened, what was learned)       │
│  2. Check: would a skill help future similar situations?         │
│  3. If yes → POST /v1/skills/manage (create or update)          │
│  4. If no → log and continue                                     │
│                                                                  │
│  Review criteria:                                                 │
│  - Action succeeded (discovery confirmed) → create/patch skill   │
│  - Action failed repeatedly → create anti-pattern skill          │
│  - Multiple agents confirmed same fact → increase confidence     │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│  SKILLS LOADER (skills_loader.py)                                │
│                                                                  │
│  Before each LLM call in context_assembler.py:                   │
│  1. Scan triggers for current situation                          │
│  2. Load matching active skills (triggers overlap)               │
│  3. Add skill content to LLM context as "learned procedures"     │
│  4. Call skills_usage.bump(name) to increment use_count          │
│                                                                  │
│  Example: If bot HP < 30% and "low_hp" trigger matches:          │
│  → Load "server-heal-strategy" skill → LLM sees "Healer at      │
│    159,193" without it being hardcoded in any Python file        │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│  SKILLS CURATOR (skills_curator.py)                              │
│                                                                  │
│  Runs every N hours (config):                                    │
│  1. Scan .usage.json for stale/archived candidates               │
│  2. For each candidate:                                          │
│     - Has it been used? No → mark stale                          │
│     - Been stale > N days? Yes → archive                         │
│     - Pinned? Skip all auto-transitions                          │
│  3. (Optional) LLM consolidation:                                │
│     - Review related stale skills                                │
│     - Merge into broader skills if applicable                    │
│     - Delete with summary if irrelevant                          │
│  4. Backup skills state before any destructive action            │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│  CONTEXT ASSEMBLER (context_assembler.py) — MODIFIED              │
│                                                                  │
│  Current:                                                        │
│  context["skills"] = operational.get("skills", {}) — EMPTY       │
│                                                                  │
│  New:                                                             │
│  context["skills"] = skills_loader.load_for_context(             │
│      situation={                                                  │
│          "bot_id": bot_id,                                        │
│          "hp_ratio": current_hp / max_hp,                         │
│          "map": current_map,                                      │
│          "zeny": current_zeny,                                    │
│          "level": current_level,                                  │
│          "in_combat": is_in_combat,                               │
│          "action_type": current_action_type,                       │
│      }                                                            │
│  )                                                                │
│  → Returns list of matching active skills as markdown blocks     │
│  → Each skill adds ~500-2000 tokens to context                    │
│  → Only skills whose triggers match the current situation         │
└─────────────────────────────────────────────────────────────────┘
```

### 3.6 CrewAI Agent Auto-Creation

```python
# In each CrewAI agent, after making a confirmed discovery:

class ProRoPlayerAgent:
    async def on_discovery(self, discovery: Discovery) -> None:
        """"Called when agent confirms a game discovery."""
        # 1. Check if skill already exists
        existing = await skills_manager.find(
            name=discovery.skill_name,
            server=discovery.server,
            domain=discovery.domain
        )
        
        # 2. Create or update
        if existing:
            await skills_manager.patch(
                name=existing.name,
                old_string=existing.version_fact,
                new_string=discovery.new_fact
            )
            skills_usage.bump(existing.name, event="patch")
        else:
            await skills_manager.create(
                name=discovery.skill_name,
                content=discovery.to_skill_md(),
                category=discovery.domain,
                metadata={"source": "crewai_discovery_agent", ...}
            )
            skills_usage.bump(discovery.skill_name, event="create")
        
        # 3. Track confidence
        skills_usage.update_confidence(
            name=discovery.skill_name,
            delta=0.05,  # Each confirmation increases confidence
        )
```

### 3.7 Discovery → Skill Flow (Example)

```
1. CrewAI navigation agent discovers: "prt_fild08 portal is at prontera (300, 180)"
2. Agent calls POST /v1/discover/tables/ingest (updates OpenKore tables)
3. Agent also calls POST /v1/skills/manage action=patch name="portal-knowledge"
   → Updates the portal skill with new portal coordinates
4. Next time bot needs to navigate to prt_fild08:
   → context_assembler loads skills matching navigation trigger
   → portal-knowledge skill says "portal at prontera (300, 180)"
   → LLM commands: move 300 180
```

---

## 4. Implementation Plan

### Phase 1: Core Files (4 files, ~600 lines total)

| File | Lines | What it does |
|------|-------|-------------|
| `skills_manager.py` | ~200 | Create/read/update/delete/patch skills with frontmatter validation |
| `skills_usage.py` | ~150 | `.usage.json` tracking: bump, state transitions, confidence |
| `skills_loader.py` | ~100 | Load matching skills into context based on trigger matching |
| `skills_curator.py` | ~150 | Background stale detection, archive, consolidation |

### Phase 2: Integration (3 files modified, 1 new)

| File | Change |
|------|--------|
| `planner/context_assembler.py` | Add `skills_loader.load_for_context(situation)` call; insert matched skills into prompt |
| `api/routers/discovery.py` | Add skill creation/update after confirming healing strategy |
| `autonomy/post_action_review.py` | NEW — trigger skill creation when agents make verified discoveries |
| `crewai/agents/base_agent.py` | Add `_maybe_create_skill()` helper that all agents inherit |

### Phase 3: API & Testing (2 new files)

| File | What it does |
|------|-------------|
| `api/routers/skills.py` | POST /v1/skills/manage, GET /v1/skills/list, GET /v1/skills/view |
| `tests/test_skills_system.py` | Test skill CRUD, lifecycle transitions, context loading |

### Phase 4: Configuration & Wiring (1 file modified)

| File | Change |
|------|--------|
| `app.py` | Register skills router; initialize skills directory; start curator background task |

---

## 5. Key Design Decisions

### 5.1 Progressive Disclosure

Like Hermes, skill loading is tiered:
- **Tier 1** (`GET /v1/skills/list`): Only name, description, tags (token-efficient, used for agent to pick skills)
- **Tier 2** (`GET /v1/skills/view?name=X`): Full SKILL.md content + linked references (loaded on demand by context_assembler)

### 5.2 Token Budget

Each skill adds ~500-2000 tokens to context. With max 3-5 matching skills per situation, budget is ~10K tokens max. Configurable via `skills.max_context_tokens` setting.

### 5.3 Write Guards

- **Foreground** (user-requested skill create/update via HTTP API): No guards, immediate effect
- **Background** (agent auto-creation from review): Must have observed the fact (not inferred). Cannot overwrite foreground-created skills.
- **Pinned**: Manual pin exempts from auto-archive. Can still be patched/edited.

### 5.4 Confidence & Reinforcement

- Each successful use: `confidence += 0.02`
- Each failed use: `confidence -= 0.05`
- Peer confirmation from multiple agents: `confidence += 0.1 per agent`
- No confirmations within 7 days: `confidence *= 0.9`
- Confidence < 0.3: Skill is candidate for consolidation review

### 5.5 Integration with Existing Systems

| Existing System | Integration |
|----------------|-------------|
| `ro_knowledge.py` axioms | Skills supplement, not replace — axioms are RO universal truths; skills are server-specific discoveries |
| `heal_resource_loader.py` | Skills replace the `knowledge.json` flat file — skills provide server-specific healing strategies |
| `discovery.py` tables cache | Skills reference and extend table data with agent-interpreted knowledge |
| `shared_learning_db.py` | Skills are the high-level "what we learned" complement to the DB's "what we observed" |
| `experience_db.py` | Skills provide the "how to" guidance that experience stats alone can't capture |

---

## 6. Verification Criteria

Before merging each phase, verify:

### Phase 1 Verification
```bash
# 1. Create a skill
curl -X POST http://127.0.0.1:18081/v1/skills/manage -H "Content-Type: application/json"   -d '{"action":"create","name":"test-heal","content":"---\nname: test-heal\ndescription: Test\n---\n# Test\nHealer at 159,193"}'

# 2. List skills
curl http://127.0.0.1:18081/v1/skills/list

# 3. View skill
curl "http://127.0.0.1:18081/v1/skills/view?name=test-heal"

# 4. Patch skill
curl -X POST http://127.0.0.1:18081/v1/skills/manage -H "Content-Type: application/json"   -d '{"action":"patch","name":"test-heal","old_string":"Healer at 159,193","new_string":"Healer at 159,193 (free)"}'

# 5. Verify .usage.json updated
cat AI_sidecar/ai_sidecar/skills/.usage.json
```

### Phase 2 Verification
```bash
# 1. Trigger heal endpoint → should include skill data in context
curl -X POST http://127.0.0.1:18081/discover/heal -H "Content-Type: application/json"   -d '{"bot_id":"test","hp":13,"hp_max":107,"map":"prontera","x":136,"y":219,"inventory":[]}'

# 2. Check response includes strategy informed by skills
#    → Expected: strategy="visit_healer_npc" (loaded from heal skill, not hardcoded)

# 3. Agent makes discovery → skill auto-created
curl http://127.0.0.1:18081/v1/skills/list | python3 -c "import sys,json; data=json.load(sys.stdin); print([s['name'] for s in data])"
# → Should include "server-heal-strategy" or similar
```

### Phase 3 Verification
```bash
# 1. Curator run: mark unused skills stale
curl -X POST http://127.0.0.1:18081/v1/skills/curate

# 2. Verify stale skills moved
cat AI_sidecar/ai_sidecar/skills/.usage.json | python3 -c "
import sys,json; d=json.load(sys.stdin)
for name, meta in d.items():
    if meta['state'] != 'active':
        print(f'{name}: {meta["state"]}')
"

# 3. Verify context assembler only loads active skills
# Inject situation → check context output for excluded stale skills
```

### Integration Test
```bash
# Full cycle: bot with 13 HP
# 1. Bridge POSTs /v1/discover/heal
# 2. Context assembler loads heal skill → strategy returned
# 3. Bridge executes move 159 193 → bot walks to Healer
# 4. Repeat: bot at Healer, HP still low → context loads heal skill → "talknpc 159 193"
# 5. Bridge executes talknpc → bot talks to Healer → HP recovered
# 6. Post-action review: confirms skill was used correctly → confidence +0.02
```

---

## 7. Files to Create (New)

```
AI_sidecar/ai_sidecar/
├── skills/
│   ├── .gitkeep                        # Empty dir init
│   └── .archive/.gitkeep               # Archive dir init
├── skills_manager.py                   # CRUD operations
├── skills_usage.py                     # Usage tracking + lifecycle
├── skills_loader.py                    # Context-aware loading
├── skills_curator.py                   # Background maintenance
├── background_review.py                # Post-action review fork
├── api/routers/skills.py               # HTTP API
└── tests/test_skills_system.py         # Tests
```

## 8. Files to Modify (Existing)

```
AI_sidecar/ai_sidecar/
├── planner/context_assembler.py        # Add skills context
├── autonomy/pdca_loop.py              # Add curator tick
├── api/routers/discovery.py            # Add skill creation on discovery
├── crewai/agents/base_agent.py         # Add _maybe_create_skill()
├── app.py                              # Register skills router, init
└── config.py                           # Add skills config section
```

---

## 9. Comparison: Hermes vs Our Implementation

| Aspect | Hermes | Our Sidecar |
|--------|--------|-------------|
| Skill dir | `~/.hermes/skills/` | `AI_sidecar/ai_sidecar/skills/` |
| Format | YAML frontmatter + markdown | Same (agentskills.io compatible) |
| Loading | Progressive disclosure (list→view) | Same |
| Triggers | `triggers` field in frontmatter | Same + programmatic trigger matching |
| Usage tracking | `.usage.json` | Same |
| Lifecycle | Active→Stale→Archived | Same |
| Curator | Weekly interval, LLM consolidation | Configurable interval, same |
| Background review | Fork agent after every turn | Post-action review after key discoveries |
| Write origin | ContextVar (foreground/background_review) | Same |
| Provenance | `skill_provenance.py` | Built into skills_manager |
| Security scan | `skills_guard.py` | Not needed (internal skills only) |
| Hub/Registry | `skills_hub.py` | Not needed (local only) |
| CrewAI integration | None (Hermes doesn't have CrewAI) | **NOVEL** — agents auto-create skills |
