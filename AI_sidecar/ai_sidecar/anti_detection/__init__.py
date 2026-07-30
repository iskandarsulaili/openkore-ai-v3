"""Anti-detection package — human-like behavior simulation for bot concealment.

This package provides:
- BehaviorEngine: Core engine for human-like play imperfection
- BridgeWiring: Connects engine output to OpenKore bridge command dispatch
- CommandPacer: Human-like timing jitter for command dispatch
- RouteHumanizer: Gaussian noise on movement waypoints
- AntiAfkEngine: Random social/activity patterns (emote, /who, inspect)
- SessionProfiler: Session-length-dependent behavior scaling
- PersonalityEngine: Per-bot personality profiles
- Legacy AntiDetection class for backward compatibility
"""

from ai_sidecar.anti_detection_legacy import AntiDetection, HumanProfile
from ai_sidecar.anti_detection.behavior_engine import (
    BehaviorEngine,
    BehaviorProfile,
    BehaviorProfileType,
    BehaviorResult,
    ContextualProfileConfig,
    HumanLikenessScorer,
    MovementDeviationConfig,
    MovementNoiseConfig,
    get_behavior_engine,
)
from ai_sidecar.anti_detection.bridge_wiring import (
    BridgeWiring,
    get_bridge_wiring,
)
from ai_sidecar.anti_detection.command_pacing import (
    CommandPacer,
    PacingProfile,
    get_command_pacer,
)
from ai_sidecar.anti_detection.route_humanizer import (
    RouteHumanizer,
    RouteHumanizerConfig,
    HumanizedRoute,
    get_route_humanizer,
)
from ai_sidecar.anti_detection.anti_afk import (
    AntiAfkEngine,
    AntiAfkConfig,
    AntiAfkAction,
    get_anti_afk_engine,
)
from ai_sidecar.anti_detection.session_profile import (
    SessionProfiler,
    SessionProfile,
    SessionProfileConfig,
    SessionPhase,
    get_session_profiler,
)

__all__ = [
    # Legacy
    "AntiDetection",
    "HumanProfile",
    # Behavior Engine
    "BehaviorEngine",
    "BehaviorProfile",
    "BehaviorProfileType",
    "BehaviorResult",
    "ContextualProfileConfig",
    "HumanLikenessScorer",
    "MovementDeviationConfig",
    "MovementNoiseConfig",
    "get_behavior_engine",
    # Bridge Wiring
    "BridgeWiring",
    "get_bridge_wiring",
    # Command Pacing
    "CommandPacer",
    "PacingProfile",
    "get_command_pacer",
    # Route Humanizer
    "RouteHumanizer",
    "RouteHumanizerConfig",
    "HumanizedRoute",
    "get_route_humanizer",
    # Anti-AFK
    "AntiAfkEngine",
    "AntiAfkConfig",
    "AntiAfkAction",
    "get_anti_afk_engine",
    # Session Profile
    "SessionProfiler",
    "SessionProfile",
    "SessionProfileConfig",
    "SessionPhase",
    "get_session_profiler",
]
