"""Anti-detection package — human-like behavior simulation for bot concealment.

This package provides the full BehaviorEngine for imperfect human-like play,
and re-exports the legacy AntiDetection class for backward compatibility.
"""

from ai_sidecar.anti_detection_legacy import AntiDetection, HumanProfile
from ai_sidecar.anti_detection.behavior_engine import (
    BehaviorEngine,
    BehaviorProfile,
    BehaviorProfileType,
    BehaviorResult,
    ContextualProfileConfig,
    HumanLikenessScorer,
    get_behavior_engine,
)

__all__ = [
    "AntiDetection",
    "BehaviorEngine",
    "BehaviorProfile",
    "BehaviorProfileType",
    "BehaviorResult",
    "ContextualProfileConfig",
    "HumanLikenessScorer",
    "HumanProfile",
    "get_behavior_engine",
]
