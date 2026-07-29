"""Combat tactics engine — per-class combat tactics, skill rotation, target selection,
opponent modeling, and job registry.

This package provides a complete combat decision-making system:

Core:
  - TacticsDispatcher: entry point, builds context, routes to tactics modules
  - TacticsContext: structured combat state for decision-making

Tactics modules (6 roles):
  - TankTactics: threat management, aggro, party protection
  - MeleeDPSTactics: burst damage, positioning
  - RangedDPSTactics: kiting, distance maintenance
  - MagicDPSTactics: elemental advantage, spell rotation
  - SupportTactics: healing, buffing, cleansing
  - HybridTactics: adaptive role based on party composition

Support modules:
  - skills: SkillRegistry (80+ skills), SkillRotationEngine
  - targeting: TargetScorer with multi-factor scoring
  - opponent_modeling: OpponentModel with spawn tracking and danger assessment
  - jobs: JobRegistry with 45+ RO class definitions

Integrates with:
  - heuristic_service.py (HeuristicAction type)
  - ro_mechanics.py (element table, monster stats, skill data)
  - domains/ base classes (BaseDomain, DomainRegistry)
"""

from ai_sidecar.domains.combat.tactics import (
    BaseTactics, TacticsContext, TargetInfo,
    TankTactics, MeleeDPSTactics, RangedDPSTactics,
    MagicDPSTactics, SupportTactics, HybridTactics,
)
from ai_sidecar.domains.combat.engine import (
    ROCombatEngine, ROMechanicsLoader, SkillInfo, SkillScore,
    CastState, get_combat_engine, get_mechanics_loader, assess_combat_engine,
)
from ai_sidecar.domains.combat.dispatcher import (
    TacticsDispatcher, get_tactics_dispatcher, assess_combat_tactics,
)
from ai_sidecar.domains.combat.skills import (
    SkillRegistry, SkillRotationEngine, SkillDef,
    Rotation, RotationStep,
    get_skill_registry, get_rotation_engine,
)
from ai_sidecar.domains.combat.targeting import (
    TargetScorer, TargetScore,
    get_target_scorer, get_tank_scorer, get_dps_scorer,
    enrich_target_with_stats, enrich_monster_list,
)
from ai_sidecar.domains.combat.opponent_modeling import (
    OpponentModel, MonsterProfile, BehaviorPrediction,
    get_opponent_model,
)
from ai_sidecar.domains.combat.jobs import (
    JobRegistry,
    get_job_registry, get_tactics_for_job,
)
from ai_sidecar.domains.combat.safety import (
    DangerPredictor,
    SafetyEvaluator,
    SafetyDomain,
    DangerAssessment,
    SafetyRecommendation,
)

__all__ = [
    # Core
    "BaseTactics", "TacticsContext", "TargetInfo",
    "TacticsDispatcher", "get_tactics_dispatcher",
    "assess_combat_tactics",

    # Engine
    "ROCombatEngine", "ROMechanicsLoader", "SkillInfo", "SkillScore",
    "CastState", "get_combat_engine", "get_mechanics_loader",
    "assess_combat_engine",

    # Tactics modules
    "TankTactics",
    "MeleeDPSTactics",
    "RangedDPSTactics",
    "MagicDPSTactics",
    "SupportTactics",
    "HybridTactics",

    # Skills
    "SkillRegistry", "SkillRotationEngine", "SkillDef",
    "Rotation", "RotationStep",
    "get_skill_registry", "get_rotation_engine",

    # Targeting
    "TargetScorer", "TargetScore",
    "get_target_scorer", "get_tank_scorer", "get_dps_scorer",
    "enrich_target_with_stats", "enrich_monster_list",

    # Opponent modeling
    "OpponentModel", "MonsterProfile", "BehaviorPrediction",
    "get_opponent_model",

    # Job registry
    "JobRegistry",
    "get_job_registry", "get_tactics_for_job",

    # Safety
    "DangerPredictor",
    "SafetyEvaluator",
    "SafetyDomain",
    "DangerAssessment",
    "SafetyRecommendation",
]
