"""Learning modules for the openkore-ai-v3 sidecar."""

from __future__ import annotations

from ai_sidecar.learning.shared_learning_db import (
    SharedLearningDB,
    SharedDeathRecord,
    SharedMVPKill,
    SharedPrice,
    get_shared_learning_db,
)
from ai_sidecar.learning.death_analysis import (
    DeathAnalyzer,
    DeathRecord,
    BehaviorAdjustment,
    get_death_analyzer,
)
from ai_sidecar.learning.strategy_optimizer import (
    StrategyOptimizer,
    get_strategy_optimizer,
)
from ai_sidecar.learning.failure_reasoning import (
    FailureReasoningEngine,
    FailureRecord,
    get_failure_reasoning_engine,
)
from ai_sidecar.learning.failure_wiring import (
    wire_failure_pipeline,
)
from ai_sidecar.learning.brain_reward_ledger import (
    BrainRewardLedger,
    BrainScore,
    BRAINS,
    get_brain_reward_ledger,
)

__all__ = [
    "SharedLearningDB",
    "SharedDeathRecord",
    "SharedMVPKill",
    "SharedPrice",
    "get_shared_learning_db",
    "DeathAnalyzer",
    "DeathRecord",
    "BehaviorAdjustment",
    "get_death_analyzer",
    "StrategyOptimizer",
    "get_strategy_optimizer",
    "FailureReasoningEngine",
    "FailureRecord",
    "get_failure_reasoning_engine",
    "BrainRewardLedger",
    "BrainScore",
    "BRAINS",
    "get_brain_reward_ledger",
    "wire_failure_pipeline",
]
