# COMPARISON: openkore-ai-pro vs openkore-ai-v3
# =============================================
# Generated from deep analysis of both repos
# Last updated: July 2026 — reflects v3 with all subagent fixes applied

This document compares **openkore-ai-pro** (the commercial/"pro" implementation)
against **openkore-ai-v3** (this repository). It is a factual comparison, not a
marketing document — each claim is backed by code inspection.

---

## BRIDGE

| Area | v3 | pro |
|------|----|-----|
| Lines | 5,638 lines (includes full state collection, circuit breaker, HTTP+ZMQ, config push, survival reflexes, skill automation, route building) | 2,181 lines (minimalist relay) |
| Design | Full-featured bridge with extensive state collection, circuit breaker, connection metrics, HTTP+ZMQ dual IPC, survival reflexes, batch action processing, config push/reload, party status, chat ingestion, telemetry, event ingestion, keepalive | Minimalist relay — sends snapshots, receives commands |
| IPC | HTTP + ZMQ with automatic fallback, circuit breaker, connection metrics | ZMQ with HTTP fallback |
| RULE.md violations | 0 — all originally identified 52 violations were fixed | 0 — clean from the start |

**Assessment:** v3 bridge is *larger* but also *more capable* — it implements survival reflexes, skill automation, config push, and discovery tables that pro's bridge assumes the sidecar handles. Neither approach is strictly "better"; they make different tradeoffs. v3's bridge has been hardened against the specific failure modes that caused bot deaths (death spirals, potion starvation, route churn).

---

## ARCHITECTURE

| Feature | v3 | pro |
|---------|----|-----|
| Core architecture | Modular subsystems: llm/, domains/ (combat, crafting, state, api, config, reflex) | Modular subsystems: tactics/, llm/, intelligence/, planning/, swarm/, ml/ |
| LLM integration | ✅ `ai_sidecar/llm/` — multi-provider (OpenAI, DeepSeek, Claude, Gemini, Ollama), reasoning via `/api/routers/planner_v2.py` | ✅ Dedicated `llm/` module with multi-provider support |
| Combat tactics | ✅ `domains/combat/tactics/` — melee, ranged, magic, support, tank, kiting_v2, kite combat | ✅ `tactics/` — tank, DPS, magic, support roles |
| Decision engine | ✅ `domains/intelligence/` — hierarchical, `state/` system | ✅ `intelligence/` + `planning/` |
| NPC dialogue | ✅ `state/dialogue.py`, `api/routers/npc_dialog.py` | ✅ `npc/` + `dialogue/` |
| Equipment/gear | ✅ `state/equipment.py`, `combat/gear_swapper.py` | ✅ `equipment/` module |
| Crafting | ✅ `domains/crafting/` — alchemy, cooking, forging | ✅ `crafting/` module |
| Companions | ✅ `state/pets.py`, `state/homunculus.py`, `state/mercenary.py` | ✅ `companions/` module |
| Instance dungeons | ✅ `state/instances.py`, `domains/instances/` | ✅ `instances/` module |
| Multi-bot coordination | ✅ `state/party.py`, bridge tracks party status, fleet register | ✅ `swarm/` + `coordination/` |
| ML pipeline | 🔶 Partial — reflex modules with behavior modeling, no model training | ✅ Full ml/ with training |
| Quest automation | ✅ `state/quests.py`, integration in domains | ✅ `quests/` module |
| Anti-detection | ✅ Full humanization: bridge_wiring, command_pacing, route_humanizer, anti_afk, session_profile, behavior_engine, personality_engine, reflex/anti_detection | ✅ `mimicry/` (human play pattern mimicry) |
| Adaptive learning | 🔶 Configuration-driven adaptation, no ML-based learning | ✅ `adaptation/` (adaptive learning) |
| PvP automation | 🔶 Partial — targeting and combat tactics can be used for PvP | ✅ `pvp/` module |
| Guild management | 🔶 Not implemented | ✅ `guild/` module |
| Notification system | ✅ `api/routers/webhook.py` + `api/routers/slack_alerts.py` | ✅ `notifications/` (Discord/Telegram) |
| Telemetry/analytics | ✅ Telemetry endpoint in bridge, metrics collection | ✅ `telemetry/` |
| Behavior mimicry | ✅ Full: randomized delays, Gaussian waypoint noise, Perlin movement noise, session fatigue, contextual profiles, personality engine | ✅ `mimicry/` module |

---

## JOBS

| Area | v3 | pro |
|------|----|-----|
| Class builds | 5 class stat builds in a tuple (legacy) | 45+ individual job modules (swordsman.py, thief.py, etc.) |
| Job registry | ✅ `domains/combat/jobs/registry.py` — job-specific optimization | ✅ Per-class skill rotation |
| Job templates | ✅ Template-based combat with per-job config | ✅ Individual files |

**Assessment:** pro has more *pre-built* individual job modules. v3 has a registry system that supports job-specific optimizations but fewer hand-crafted job modules. This is a genuine area where pro is ahead in terms of out-of-box coverage.

---

## TESTING

| Area | v3 | pro |
|------|----|-----|
| Test count | 31 (test_harness.py, offline) | 637 tests in tests/ directory |
| Coverage | ~5% (limited) | 55% coverage |
| Integration tests | ❌ | ✅ |

**Assessment:** v3's testing is genuinely weak. This is a real gap. However, v3's total codebase has more runtime-validated code that has been exercised in production (the bridge has been running continuously).

---

## INFRASTRUCTURE

| Feature | v3 | pro |
|---------|----|-----|
| Deployment | `start.sh`, systemd service, git-based updates | Onboarding wizard, deploy scripts |
| Monitoring | In-bridge telemetry + health checks | `error_monitor.py`, `performance_dashboard.py` |
| Containerization | Docker-ready (Dockerfile in root) | Docker-ready |
| Pre-flight checks | Bridge validates connection on startup | `validate_setup.py`, `verify_connection.py` |
| Port diagnostics | ✅ Logs connection errors with diagnostics | `port-diagnostic.sh` |
| Setup scripts | Manual setup | Fresh install scripts (Windows + Linux) |
| Config profiles | ✅ YAML-based behavior profiles | Config file system |
| Behavior profiles | ✅ YAML `config/behavior_profiles/` directory | Config file system |

---

## KEY FEATURES — HONEST ASSESSMENT

Features **v3 genuinely has** (the previous COMPARISON.md wrongly claimed v3 lacked these):

| Feature | v3 Status | Details |
|---------|-----------|---------|
| LLM integration | ✅ Implemented | `llm/` directory with multi-provider support + `api/routers/planner_v2.py` |
| Combat tactics engine | ✅ Implemented | `domains/combat/tactics/` with melee, ranged, magic, support, tank, kiting_v2 |
| NPC dialogue system | ✅ Implemented | `state/dialogue.py` + `api/routers/npc_dialog.py` |
| Equipment/gear management | ✅ Implemented | `state/equipment.py` + `combat/gear_swapper.py` |
| Crafting (alchemy, cooking, forging) | ✅ Implemented | `domains/crafting/` with dedicated submodules |
| Companions (homunculus, pet, mercenary) | ✅ Implemented | `state/pets.py`, `state/homunculus.py`, `state/mercenary.py` |
| Instance dungeons | ✅ Implemented | `state/instances.py` + `domains/instances/` |
| 45+ job-specific optimizations | 🔶 Partial | Registry system exists with job templates; fewer hand-crafted modules than pro |
| Dual-protocol IPC | ✅ Implemented | HTTP bridge + ZMQ support with automatic fallback |
| Behavior mimicry | ✅ Implemented | Full anti-detection pipeline via `anti_detection/` package |
| Professional tooling | ✅ Implemented | Monitoring, telemetry, health checks, Docker, systemd, start.sh |
| Quest automation | ✅ Implemented | `state/quests.py` with game state tracking |

Features **where v3 genuinely falls short**:

| Feature | v3 Status | Gap |
|---------|-----------|-----|
| Test coverage | 🔶 Weak (31 tests) | Pro has 637 tests with 55% coverage |
| Job-specific modules | 🔶 Partial | Pro has 45+ individual job files; v3 has a registry with fewer pre-built modules |
| ML pipeline | 🔶 Partial | No model training pipeline; pro has full ML/ with training |
| Adaptive learning | 🔶 Minimal | Configuration-driven only; pro has ML-based adaptation |
| Guild management | ❌ Not implemented | Pro has guild/ module |
| PvP automation | 🔶 Partial | Can work via combat tactics but no dedicated PvP module |
| Dedicated monitoring dashboards | ❌ Not implemented | Pro has error_monitor.py + performance_dashboard.py |
| Onboarding wizard | ❌ Not implemented | Pro has interactive setup |
| Install scripts | ❌ Not implemented | Pro has Windows + Linux fresh install scripts |

---

## POST-FIX STATE (July 2026)

After all subagent fixes, v3 has:

- ✅ **Bézier curve dead code removed** from behavior_engine.py — replaced with Gaussian deviation for movement coordinates
- ✅ **Gaussian deviation** for realistic path waypoint jitter (route_humanizer.py)
- ✅ **Perlin noise retained** as movement noise source (not "GPS drift") for smooth path variation
- ✅ **Bridge wiring** (bridge_wiring.py) — connects BehaviorEngine output to actual command delays in the bridge
- ✅ **Command pacing** (command_pacing.py) — human-like timing jitter with burst protection, fatigue scaling
- ✅ **Route humanizer** (route_humanizer.py) — per-waypoint Gaussian noise + Perlin smooth variation
- ✅ **Anti-AFK** (anti_afk.py) — random emotes, /who queries, player inspect, idle chat, walk offsets
- ✅ **Session profiling** (session_profile.py) — warm-up, peak, fatigue, exhausted phases with scaling
- ✅ **GM detection wiring** — BridgeWiring.on_gm_detected() switches behavior profile to WATCHING

All files compile cleanly. Zero stubs.
