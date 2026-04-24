from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from time import perf_counter

from ai_sidecar.config import settings
from ai_sidecar.contracts.actions import ActionPriorityTier, ActionProposal, ActionStatus
from ai_sidecar.fleet.sync_client import FleetSyncClient
from ai_sidecar.runtime.action_queue import ActionQueue
from ai_sidecar.fleet.constraint_ingestion import ConstraintIngestionState

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AdmissionResult:
    admitted: bool
    reason: str
    action_id: str | None
    status: ActionStatus | None = None


class ActionArbiter:
    def __init__(
        self,
        queue: ActionQueue,
        fleet_client: FleetSyncClient | None = None,
        constraint_state: ConstraintIngestionState | None = None,
    ) -> None:
        self._queue = queue
        self._fleet_client = fleet_client
        self._constraint_state = constraint_state

    async def admit(self, proposal: ActionProposal, *, bot_id: str | None = None) -> AdmissionResult:
        return self._admit_impl(proposal, bot_id=bot_id)

    def admit_sync(self, proposal: ActionProposal, *, bot_id: str | None = None) -> AdmissionResult:
        return self._admit_impl(proposal, bot_id=bot_id)

    def get_queue_summary(self) -> dict[str, object]:
        snapshot = self._queue.snapshot()
        by_source: dict[str, int] = {}
        by_priority: dict[str, int] = {}
        by_status: dict[str, int] = {}
        by_bot: dict[str, int] = {}
        total = 0
        for bot_id, actions in snapshot.items():
            by_bot[bot_id] = len(actions)
            for queued in actions:
                total += 1
                source = queued.proposal.source or "manual"
                by_source[source] = by_source.get(source, 0) + 1
                priority = queued.proposal.priority_tier.value
                by_priority[priority] = by_priority.get(priority, 0) + 1
                status = queued.status.value
                by_status[status] = by_status.get(status, 0) + 1
        return {
            "total": total,
            "by_source": by_source,
            "by_priority": by_priority,
            "by_status": by_status,
            "by_bot": by_bot,
        }

    def _admit_impl(self, proposal: ActionProposal, *, bot_id: str | None = None) -> AdmissionResult:
        started = perf_counter()
        resolved_bot_id = self._resolve_bot_id(proposal, bot_id)
        if not resolved_bot_id:
            logger.warning(
                "action_admission_missing_bot_id",
                extra={"event": "action_admission_missing_bot_id", "action_id": proposal.action_id},
            )
            return AdmissionResult(
                admitted=False,
                reason="missing_bot_id",
                action_id=proposal.action_id,
                status=ActionStatus.dropped,
            )

        proposal = self._normalize_proposal(proposal)
        logger.info(
            "action_admission_received",
            extra={
                "event": "action_admission_received",
                "bot_id": resolved_bot_id,
                "action_id": proposal.action_id,
                "source": proposal.source,
                "priority": proposal.priority_tier.value,
            },
        )

        if proposal.preconditions:
            logger.warning(
                "action_admission_preconditions_unmet",
                extra={
                    "event": "action_admission_preconditions_unmet",
                    "bot_id": resolved_bot_id,
                    "action_id": proposal.action_id,
                    "preconditions": list(proposal.preconditions),
                },
            )
            return AdmissionResult(
                admitted=False,
                reason="preconditions_unmet",
                action_id=proposal.action_id,
                status=ActionStatus.dropped,
            )

        if proposal.lease_id:
            lease_id = str(proposal.lease_id).strip()
            if not lease_id:
                logger.warning(
                    "action_admission_invalid_lease",
                    extra={
                        "event": "action_admission_invalid_lease",
                        "bot_id": resolved_bot_id,
                        "action_id": proposal.action_id,
                    },
                )
                return AdmissionResult(
                    admitted=False,
                    reason="fleet_lease_invalid",
                    action_id=proposal.action_id,
                    status=ActionStatus.dropped,
                )
            if self._fleet_client is None or not self._fleet_client.enabled:
                logger.warning(
                    "action_admission_fleet_unavailable",
                    extra={
                        "event": "action_admission_fleet_unavailable",
                        "bot_id": resolved_bot_id,
                        "action_id": proposal.action_id,
                        "lease_id": lease_id,
                    },
                )
                return AdmissionResult(
                    admitted=False,
                    reason="fleet_lease_unavailable",
                    action_id=proposal.action_id,
                    status=ActionStatus.dropped,
                )

        if proposal.conflict_key:
            conflicts = self._conflicts_for_bot(resolved_bot_id, proposal.conflict_key)
            logger.info(
                "action_admission_conflict_check",
                extra={
                    "event": "action_admission_conflict_check",
                    "bot_id": resolved_bot_id,
                    "action_id": proposal.action_id,
                    "conflict_key": proposal.conflict_key,
                    "conflict_count": len(conflicts),
                },
            )

        if self._constraint_state is not None:
            fleet_result = self._check_fleet_constraints(proposal, resolved_bot_id)
            if not fleet_result.admitted:
                logger.warning(
                    "action_admission_fleet_rejected",
                    extra={
                        "event": "action_admission_fleet_rejected",
                        "bot_id": resolved_bot_id,
                        "action_id": proposal.action_id,
                        "reason": fleet_result.reason,
                    },
                )
                return fleet_result

        elapsed_ms = (perf_counter() - started) * 1000.0
        budget_ms = self._resolve_latency_budget_ms(proposal)
        logger.info(
            "action_admission_latency_budget",
            extra={
                "event": "action_admission_latency_budget",
                "bot_id": resolved_bot_id,
                "action_id": proposal.action_id,
                "elapsed_ms": elapsed_ms,
                "budget_ms": budget_ms,
                "source": proposal.source,
            },
        )
        if elapsed_ms > budget_ms:
            logger.warning(
                "action_admission_latency_exceeded",
                extra={
                    "event": "action_admission_latency_exceeded",
                    "bot_id": resolved_bot_id,
                    "action_id": proposal.action_id,
                    "elapsed_ms": elapsed_ms,
                    "budget_ms": budget_ms,
                },
            )
            return AdmissionResult(
                admitted=False,
                reason="latency_budget_exceeded",
                action_id=proposal.action_id,
                status=ActionStatus.dropped,
            )

        accepted, status, action_id, reason = self._queue.enqueue(resolved_bot_id, proposal)
        logger.info(
            "action_admission_queue_result",
            extra={
                "event": "action_admission_queue_result",
                "bot_id": resolved_bot_id,
                "action_id": action_id,
                "requested_action_id": proposal.action_id,
                "accepted": accepted,
                "status": status.value,
                "reason": reason,
            },
        )
        return AdmissionResult(
            admitted=accepted,
            reason=reason,
            action_id=action_id,
            status=status,
        )

    def _resolve_bot_id(self, proposal: ActionProposal, bot_id: str | None) -> str | None:
        if bot_id:
            return str(bot_id).strip() or None
        metadata_bot_id = proposal.metadata.get("bot_id") if isinstance(proposal.metadata, dict) else None
        if metadata_bot_id:
            return str(metadata_bot_id).strip() or None
        return None

    def _normalize_proposal(self, proposal: ActionProposal) -> ActionProposal:
        updates: dict[str, object] = {}
        metadata = dict(proposal.metadata or {})

        if proposal.source == "manual":
            derived_source = self._map_source(metadata.get("source"))
            if derived_source is not None:
                updates["source"] = derived_source

        if proposal.latency_budget_ms is None:
            latency_raw = metadata.get("latency_budget_ms")
            if isinstance(latency_raw, (int, float)):
                updates["latency_budget_ms"] = int(latency_raw)

        if not proposal.preconditions:
            preconditions_raw = metadata.get("preconditions")
            if isinstance(preconditions_raw, list):
                updates["preconditions"] = [str(item) for item in preconditions_raw if str(item).strip()]

        if proposal.rollback_action is None:
            rollback_raw = metadata.get("rollback_action") or metadata.get("rollback_hint")
            if rollback_raw:
                updates["rollback_action"] = str(rollback_raw)[:256]

        if proposal.lease_id is None:
            lease_raw = metadata.get("lease_id")
            if lease_raw:
                updates["lease_id"] = str(lease_raw)[:128]

        if "source" not in updates and proposal.source == "manual" and proposal.priority_tier == ActionPriorityTier.strategic:
            updates["source"] = "planner"

        if proposal.ttl_seconds is None:
            ttl_raw = metadata.get("ttl_seconds")
            if isinstance(ttl_raw, (int, float)):
                updates["ttl_seconds"] = int(ttl_raw)

        if proposal.ttl_seconds is not None:
            ttl_seconds = max(1, int(proposal.ttl_seconds))
            expires_at = proposal.created_at + timedelta(seconds=ttl_seconds)
            if proposal.expires_at > expires_at:
                updates["expires_at"] = expires_at

        return proposal.model_copy(update=updates) if updates else proposal

    def _map_source(self, raw: object) -> str | None:
        value = str(raw or "").strip().lower()
        if not value:
            return None
        if value in {"reflex", "planner", "crewai", "ml", "fleet", "manual"}:
            return value
        if "reflex" in value:
            return "reflex"
        if "planner" in value:
            return "planner"
        if "pdca" in value:
            return "planner"
        if "crew" in value:
            return "crewai"
        if "fleet" in value:
            return "fleet"
        if "ml" in value or "model" in value:
            return "ml"
        return "manual"

    def _resolve_latency_budget_ms(self, proposal: ActionProposal) -> float:
        if proposal.latency_budget_ms is not None:
            return max(0.0, float(proposal.latency_budget_ms))
        if proposal.source == "reflex":
            return max(0.0, float(settings.reflex_latency_budget_ms))
        return max(0.0, float(settings.latency_budget_ms))

    def _conflicts_for_bot(self, bot_id: str, conflict_key: str) -> list[ActionProposal]:
        snapshot = self._queue.snapshot().get(bot_id, [])
        out: list[ActionProposal] = []
        for queued in snapshot:
            if queued.status != ActionStatus.queued:
                continue
            if queued.proposal.conflict_key != conflict_key:
                continue
            out.append(queued.proposal)
        return out

    def _check_fleet_constraints(self, proposal: ActionProposal, bot_id: str) -> AdmissionResult:
        """Check action against fleet blackboard constraints."""
        state = self._constraint_state
        if state is None:
            return AdmissionResult(admitted=True, reason="fleet_constraints_unavailable", action_id=proposal.action_id)

        metadata = proposal.metadata if isinstance(proposal.metadata, dict) else {}
        target_map = None
        if isinstance(metadata.get("map"), str):
            target_map = str(metadata.get("map") or "")
        if not target_map and isinstance(metadata.get("target_map"), str):
            target_map = str(metadata.get("target_map") or "")

        if target_map:
            if state.is_zone_claimed(target_map, bot_id):
                claimant = state.get_zone_claim(target_map)
                claimed_by = claimant.claimed_by if claimant is not None else "unknown"
                return AdmissionResult(
                    admitted=False,
                    reason=f"zone_claimed:{target_map}:{claimed_by}",
                    action_id=proposal.action_id,
                    status=ActionStatus.dropped,
                )

        if proposal.lease_id:
            lease = state.get_task_lease(str(proposal.lease_id))
            if lease is None or lease.status != "active":
                return AdmissionResult(
                    admitted=False,
                    reason=f"lease_inactive:{proposal.lease_id}",
                    action_id=proposal.action_id,
                    status=ActionStatus.dropped,
                )
            if lease.assigned_to and lease.assigned_to != bot_id:
                return AdmissionResult(
                    admitted=False,
                    reason=f"lease_wrong_owner:{proposal.lease_id}:{lease.assigned_to}",
                    action_id=proposal.action_id,
                    status=ActionStatus.dropped,
                )

        doctrine = state.get_doctrine()
        if doctrine is not None and target_map:
            for rule in doctrine.rules:
                if rule.startswith("avoid_map:"):
                    avoid_map = rule.split(":", 1)[1]
                    if target_map == avoid_map:
                        return AdmissionResult(
                            admitted=False,
                            reason=f"doctrine_forbid_map:{avoid_map}",
                            action_id=proposal.action_id,
                            status=ActionStatus.dropped,
                        )

        if target_map:
            threats = state.get_active_threats(target_map)
            high_threats = [item for item in threats if item.severity >= 4]
            if high_threats:
                logger.warning(
                    "action_rejected_high_threat",
                    extra={
                        "event": "action_rejected_high_threat",
                        "map": target_map,
                        "threats": len(high_threats),
                        "action_id": proposal.action_id,
                    },
                )
                return AdmissionResult(
                    admitted=False,
                    reason=f"high_threat:{target_map}",
                    action_id=proposal.action_id,
                    status=ActionStatus.dropped,
                )

        return AdmissionResult(admitted=True, reason="fleet_ok", action_id=proposal.action_id)
