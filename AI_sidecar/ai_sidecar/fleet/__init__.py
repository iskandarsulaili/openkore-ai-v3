"""Fleet synchronization, coordination, party management, and self-learning services."""

from ai_sidecar.fleet.constraint_ingestion import ConstraintIngestionState
from ai_sidecar.fleet.conflict_resolver import FleetConflictResolver
from ai_sidecar.fleet.outcome_reporter import OutcomeReporter
from ai_sidecar.fleet.role_manager import RoleManager
from ai_sidecar.fleet.sync_client import FleetSyncClient
from ai_sidecar.fleet.coordinator import (
    FleetCoordinatorService,
    BotFleetState,
    FleetMessage,
    RoleMetrics,
    RoleType,
)
from ai_sidecar.fleet.party_coordinator import PartyCoordinator, CoordinationAction
from ai_sidecar.fleet.self_learning import (
    SelfLearningSystem,
    PerformanceRecord,
    SkillEffectiveness,
    ItemEffectiveness,
)

__all__ = [
    # Legacy
    "FleetSyncClient",
    "ConstraintIngestionState",
    "OutcomeReporter",
    "RoleManager",
    "FleetConflictResolver",
    # New fleet coordination
    "FleetCoordinatorService",
    "BotFleetState",
    "FleetMessage",
    "RoleMetrics",
    "RoleType",
    # Party coordination
    "PartyCoordinator",
    "CoordinationAction",
    # Self-learning
    "SelfLearningSystem",
    "PerformanceRecord",
    "SkillEffectiveness",
    "ItemEffectiveness",
]
