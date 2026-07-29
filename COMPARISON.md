# COMPARISON: openkore-ai-pro vs openkore-ai-v3
# Generated from deep analysis of both repos

BRIDGE:
  v3:  5,638 lines (bloated bridge with too much logic)
  pro: 2,181 lines (clean, focused, ZMQ + HTTP fallback)
  
  v3 bridge - 52 RULE.md violations (fixed), still has config fallbacks
  pro bridge - minimalist relay, no config overrides from the start

ARCHITECTURE:
  v3:  Hand-coded heuristic in one monolithic heuristic_service.py (3,718 lines)
  pro:  Modular subsystems broken into specialized modules:
       - tactics/        (combat tactics - tank, DPS, magic, etc.)
       - llm/            (multi-provider LLM: OpenAI, Azure, Claude, DeepSeek)
       - intelligence/   (decision engine hierarchy)
       - planning/       (strategic planning)
       - swarm/          (multi-bot coordination)
       - ml/             (machine learning models)
       - combat/         (critical, combos, race property, targeting)
       - quests/         (quest automation)
       - npc/            (NPC interaction)
       - crafting/       (alchemy, cooking, forging)
       - equipment/      (gear management)
       - consumables/    (buff/recovery management)
       - companions/     (homunculus/pet/mercenary)
       - progression/    (character lifecycle)
       - pvp/            (PvP tactics)
       - instances/      (instance dungeon support)
       - guild/          (guild management)
       - notifications/  (Discord/Telegram)
       - dialogue/       (dialogue parsing)
       - opponent_modeling/ (ML monster behavior prediction)
       - adaptation/     (adaptive learning)
       - mimicry/        (human play pattern mimicry)
       - coordination/   (bot-to-bot coordination)

JOBS:
  v3:  5 class stat builds in a tuple
  pro: 45+ individual job modules (swordsman.py, thief.py, paladin.py, etc.)

TESTING:
  v3:  1 test_harness.py (31 tests, offline only)
  pro: 637 tests (in tests/ directory), 55% coverage, integration tests

INFRASTRUCTURE:
  v3:  start.sh, git, basic
  pro:  
    - Onboarding wizard (interactive setup)
    - Monitoring dashboards (error_monitor.py, performance_dashboard.py)
    - Deployment scripts (configure-discord.py, configure-telegram.py)
    - Pre-flight checks (validate_setup.py, verify_connection.py)
    - Port diagnostics (port-diagnostic.sh)
    - Fresh install scripts (Windows + Linux)
    - Docker-ready environment

KEY FEATURES v3 LACKS:
  1. Combat tactics engine (tank, DPS, magic, support roles with actual behavior differences)
  2. LLM integration (multi-provider reasoning, not just a single /team-synergy endpoint)
  3. Swarm intelligence (coordinated multi-bot formations, not just party invite)
  4. ML pipeline (actual model training, not hand-coded rules)
  5. Opponent modeling (predict monster behavior, not just mon_control -1)
  6. Quest automation (accept → complete → turn in)
  7. NPC dialogue system (parse dialogue trees, not hardcoded c r1 c)
  8. Equipment/gear management (auto-switch weapons for element advantage)
  9. Crafting (alchemy, cooking, forging for consumables/income)
  10. Companions (homunculus, pet, mercenary, mount management)
  11. Instance dungeons (enter, complete, exit instances)
  12. Guild management (auto-guild activities)
  13. PvP automation (arena tactics)
  14. Notification system (Discord/Telegram alerts)
  15. Behavior mimicry (randomized delays, human-like movement)
  16. Adaptive learning (learn from success/failure over time)
  17. Dialogue parsing (read NPC dialogue for quest decisions)
  18. 45+ job-specific optimizations (per-class skill rotation, stat distribution)
  19. Dual-protocol IPC (ZMQ auto-fallback to HTTP)
  20. Telemetry/analytics (collect performance data for optimization)
  21. 637 test suite with 55% coverage
  22. Professional tooling (onboarding wizard, monitoring, deployment)
