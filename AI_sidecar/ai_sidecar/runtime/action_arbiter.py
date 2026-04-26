from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from time import perf_counter

from ai_sidecar.config import settings
from ai_sidecar.contracts.actions import ActionPriorityTier, ActionProposal, ActionStatus
from ai_sidecar.contracts.state import BotStateSnapshot
from ai_sidecar.fleet.sync_client import FleetSyncClient
from ai_sidecar.runtime.action_queue import ActionQueue
from ai_sidecar.fleet.constraint_ingestion import ConstraintIngestionState
from ai_sidecar.runtime.snapshot_cache import SnapshotCache

logger = logging.getLogger(__name__)

_PRECONDITION_SNAPSHOT_MISSING_WARN_EVERY = 5


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
        snapshot_cache: SnapshotCache | None = None,
    ) -> None:
        self._queue = queue
        self._fleet_client = fleet_client
        self._constraint_state = constraint_state
        self._snapshot_cache = snapshot_cache
        self._precondition_snapshot_missing_counts: dict[str, int] = {}

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

        random_walk_guard = self._validate_random_walk_guard(proposal, bot_id=resolved_bot_id)
        if random_walk_guard is not None:
            return random_walk_guard

        precondition_result = self._evaluate_preconditions(proposal, bot_id=resolved_bot_id)
        if precondition_result is not None:
            return precondition_result

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

    def _validate_random_walk_guard(self, proposal: ActionProposal, *, bot_id: str) -> AdmissionResult | None:
        if not self._requires_target_scan_guard(proposal):
            return None

        required_precondition = "scan.targets_absent"
        preconditions = {str(item).strip().lower() for item in proposal.preconditions}
        if required_precondition not in preconditions:
            logger.warning(
                "action_admission_random_walk_missing_scan_precondition",
                extra={
                    "event": "action_admission_random_walk_missing_scan_precondition",
                    "bot_id": bot_id,
                    "action_id": proposal.action_id,
                    "command": proposal.command,
                },
            )
            return AdmissionResult(
                admitted=False,
                reason="random_walk_requires_target_scan",
                action_id=proposal.action_id,
                status=ActionStatus.dropped,
            )

        metadata = proposal.metadata if isinstance(proposal.metadata, dict) else {}
        target_scan = metadata.get("target_scan") if isinstance(metadata.get("target_scan"), dict) else {}
        if self._coerce_bool(target_scan.get("targets_found")) is True:
            logger.warning(
                "action_admission_random_walk_scan_failed",
                extra={
                    "event": "action_admission_random_walk_scan_failed",
                    "bot_id": bot_id,
                    "action_id": proposal.action_id,
                    "command": proposal.command,
                    "target_scan": target_scan,
                },
            )
            return AdmissionResult(
                admitted=False,
                reason="random_walk_target_scan_failed",
                action_id=proposal.action_id,
                status=ActionStatus.dropped,
            )

        if self._coerce_bool(metadata.get("targets_present")) is True:
            logger.warning(
                "action_admission_random_walk_targets_present",
                extra={
                    "event": "action_admission_random_walk_targets_present",
                    "bot_id": bot_id,
                    "action_id": proposal.action_id,
                },
            )
            return AdmissionResult(
                admitted=False,
                reason="random_walk_target_scan_failed",
                action_id=proposal.action_id,
                status=ActionStatus.dropped,
            )

        return None

    def _evaluate_preconditions(self, proposal: ActionProposal, *, bot_id: str) -> AdmissionResult | None:
        if not proposal.preconditions:
            return None

        snapshot_cache = self._snapshot_cache
        if snapshot_cache is None:
            logger.warning(
                "action_admission_precondition_snapshot_cache_unavailable",
                extra={
                    "event": "action_admission_precondition_snapshot_cache_unavailable",
                    "bot_id": bot_id,
                    "action_id": proposal.action_id,
                    "preconditions": list(proposal.preconditions),
                },
            )
            return None

        snapshot = snapshot_cache.get(bot_id)
        if snapshot is None:
            miss_count = self._precondition_snapshot_missing_counts.get(bot_id, 0) + 1
            self._precondition_snapshot_missing_counts[bot_id] = miss_count
            log_extra = {
                "event": "action_admission_precondition_snapshot_missing",
                "bot_id": bot_id,
                "action_id": proposal.action_id,
                "preconditions": list(proposal.preconditions),
                "miss_count": miss_count,
                "warn_every": _PRECONDITION_SNAPSHOT_MISSING_WARN_EVERY,
            }
            if miss_count == 1:
                logger.info("action_admission_precondition_snapshot_missing", extra=log_extra)
            elif miss_count % _PRECONDITION_SNAPSHOT_MISSING_WARN_EVERY == 0:
                logger.warning("action_admission_precondition_snapshot_missing", extra=log_extra)
            else:
                logger.debug("action_admission_precondition_snapshot_missing", extra=log_extra)
            return None

        self._precondition_snapshot_missing_counts.pop(bot_id, None)

        for item in proposal.preconditions:
            name = str(item).strip()
            if not name:
                continue
            passed, detail, unknown = self._evaluate_precondition(name, snapshot)
            logger.info(
                "action_admission_precondition_evaluated",
                extra={
                    "event": "action_admission_precondition_evaluated",
                    "bot_id": bot_id,
                    "action_id": proposal.action_id,
                    "precondition": name,
                    "passed": passed,
                    "detail": detail,
                },
            )
            if unknown:
                logger.warning(
                    "action_admission_precondition_unknown",
                    extra={
                        "event": "action_admission_precondition_unknown",
                        "bot_id": bot_id,
                        "action_id": proposal.action_id,
                        "precondition": name,
                    },
                )
                continue
            if not passed:
                logger.warning(
                    "action_admission_precondition_failed",
                    extra={
                        "event": "action_admission_precondition_failed",
                        "bot_id": bot_id,
                        "action_id": proposal.action_id,
                        "precondition": name,
                        "detail": detail,
                    },
                )
                return AdmissionResult(
                    admitted=False,
                    reason=f"precondition_failed:{name}",
                    action_id=proposal.action_id,
                    status=ActionStatus.dropped,
                )

        return None

    def _evaluate_precondition(self, name: str, snapshot: BotStateSnapshot) -> tuple[bool, str, bool]:
        raw = snapshot.raw if isinstance(snapshot.raw, dict) else {}
        map_name = self._resolve_map_name(snapshot, raw)
        status, status_source = self._resolve_status(snapshot, raw)

        if name == "navigation.ready":
            if status == "dead":
                return False, f"status=dead source={status_source}", False
            if not map_name:
                return True, "map_missing", False
            return True, f"map={map_name} status={status or 'unknown'}", False

        if name == "combat.allowed":
            is_alive = None
            if status:
                is_alive = status == "alive"
            else:
                hp = snapshot.vitals.hp
                if hp is not None:
                    is_alive = hp > 0
            is_town = self._resolve_is_town(map_name, raw)
            if is_alive is False:
                return False, f"status={status or 'dead'}", False
            if is_town is True:
                return False, f"map_town={map_name or 'unknown'}", False
            if is_alive is None or is_town is None:
                return True, "insufficient_data", False
            return True, f"map={map_name or 'unknown'}", False

        if name == "inventory.can_loot":
            weight = self._coerce_float(snapshot.vitals.weight)
            max_weight = self._coerce_float(snapshot.vitals.weight_max)
            if weight is None:
                weight = self._coerce_float(raw.get("weight"))
            if max_weight is None:
                max_weight = self._coerce_float(raw.get("weight_max") or raw.get("max_weight"))
            if weight is None or max_weight is None:
                return True, "weight_missing", False
            if max_weight <= 0:
                return True, "weight_max_invalid", False
            return weight < max_weight, f"weight={weight} max_weight={max_weight}", False

        if name == "vitals.safe_to_rest":
            hp = self._coerce_float(snapshot.vitals.hp)
            hp_max = self._coerce_float(snapshot.vitals.hp_max)
            sp = self._coerce_float(snapshot.vitals.sp)
            sp_max = self._coerce_float(snapshot.vitals.sp_max)
            hp_ratio = (hp / hp_max) if hp is not None and hp_max and hp_max > 0 else None
            sp_ratio = (sp / sp_max) if sp is not None and sp_max and sp_max > 0 else None
            if hp_ratio is None and sp_ratio is None:
                return True, "vitals_missing", False
            needs_rest = bool((hp_ratio is not None and hp_ratio < 0.7) or (sp_ratio is not None and sp_ratio < 0.5))
            return needs_rest, f"hp_ratio={hp_ratio} sp_ratio={sp_ratio}", False

        if name == "npc.available":
            npc_relationships = len(snapshot.npc_relationships)
            npc_actors = sum(1 for actor in snapshot.actors if (actor.actor_type or "").lower() == "npc")
            has_npc = npc_relationships > 0 or npc_actors > 0
            return has_npc, f"npc_relationships={npc_relationships} npc_actors={npc_actors}", False

        if name == "economy.safe":
            monsters = self._coerce_int(raw.get("monsters_around"))
            if monsters is None:
                monsters = sum(1 for actor in snapshot.actors if (actor.actor_type or "").lower() == "monster")
            if monsters is None:
                return True, "monsters_unknown", False
            return monsters < 3, f"monsters_around={monsters}", False

        if name == "progression.skill_points_available":
            skill_points = self._coerce_int(snapshot.progression.skill_points)
            if skill_points is None:
                return True, "skill_points_missing", False
            return skill_points > 0, f"skill_points={skill_points}", False

        if name == "social.allowed":
            party_enabled = raw.get("party_enabled")
            if isinstance(party_enabled, bool):
                return party_enabled, f"party_enabled={party_enabled}", False
            party_cfg = raw.get("party") if isinstance(raw.get("party"), dict) else None
            if isinstance(party_cfg, dict) and isinstance(party_cfg.get("enabled"), bool):
                enabled = bool(party_cfg.get("enabled"))
                return enabled, f"party_enabled={enabled}", False
            return True, "party_config_missing", False

        if name == "inventory.can_equip":
            equipment_slots = raw.get("equipment_slots")
            if isinstance(equipment_slots, dict):
                has_empty = any(not value for value in equipment_slots.values())
                return has_empty, f"equipment_slots_empty={has_empty}", False
            if isinstance(equipment_slots, list):
                has_empty = any(not value for value in equipment_slots)
                return has_empty, f"equipment_slots_empty={has_empty}", False
            unequipped = sum(
                1
                for item in snapshot.inventory_items
                if (item.category or "").lower() in {"equipment", "armor", "weapon", "gear"} and not item.equipped
            )
            if unequipped:
                return True, f"unequipped_items={unequipped}", False
            return True, "equipment_data_missing", False

        if name == "social.party_ready":
            party_nearby = self._coerce_int(raw.get("party_nearby") or raw.get("party_members_nearby"))
            if party_nearby is None:
                party_nearby = sum(1 for actor in snapshot.actors if (actor.relation or "").lower() == "party")
            if party_nearby is None:
                return True, "party_nearby_unknown", False
            return party_nearby > 0, f"party_nearby={party_nearby}", False

        if name == "crafting.ready":
            materials_count = self._coerce_int(
                raw.get("crafting_materials_count")
                or (len(raw.get("crafting_materials")) if isinstance(raw.get("crafting_materials"), list) else None)
            )
            if materials_count is None:
                materials_count = sum(
                    1
                    for item in snapshot.inventory_items
                    if (item.category or "").lower() in {"material", "craft", "ingredient", "resource"}
                    and item.quantity > 0
                )
            if materials_count is None:
                return True, "materials_unknown", False
            return materials_count > 0, f"materials_count={materials_count}", False

        if name == "scan.targets_absent":
            target_id = str(raw.get("target_id") or snapshot.combat.target_id or "").strip()
            monsters = self._coerce_int(raw.get("monsters_around"))
            if monsters is None:
                monsters = sum(1 for actor in snapshot.actors if (actor.actor_type or "").lower() == "monster")
            targets_found = bool(target_id) or bool(monsters and monsters > 0)
            return (
                not targets_found,
                f"targets_found={targets_found} target_id={target_id or 'none'} monsters_around={monsters}",
                False,
            )

        return True, "unknown_precondition", True

    def _requires_target_scan_guard(self, proposal: ActionProposal) -> bool:
        command = str(proposal.command or "").strip().lower()
        metadata = proposal.metadata if isinstance(proposal.metadata, dict) else {}
        if self._coerce_bool(metadata.get("seek_only_random_walk")) is True:
            return True
        if self._coerce_bool(metadata.get("target_scan_required")) is True:
            return True
        if "random_walk" in command or "randomwalk" in command:
            return True
        if command.startswith("route_randomwalk"):
            return True
        return False

    def _resolve_map_name(self, snapshot: BotStateSnapshot, raw: dict[str, object]) -> str | None:
        map_name = snapshot.position.map or raw.get("map") or raw.get("map_name")
        map_name = str(map_name or "").strip()
        return map_name or None

    def _resolve_status(self, snapshot: BotStateSnapshot, raw: dict[str, object]) -> tuple[str, str]:
        for key in ("status", "liveness_state", "state"):
            value = raw.get(key)
            if value:
                return str(value).strip().lower(), f"raw:{key}"
        hp = snapshot.vitals.hp
        hp_max = snapshot.vitals.hp_max
        if hp is not None and hp_max is not None and hp_max > 0:
            return ("dead" if hp <= 0 else "alive"), "vitals"
        if hp is not None and hp <= 0:
            return "dead", "vitals"
        return "", "unknown"

    def _resolve_is_town(self, map_name: str | None, raw: dict[str, object]) -> bool | None:
        if isinstance(raw.get("map_is_town"), bool):
            return bool(raw.get("map_is_town"))
        if isinstance(raw.get("is_town"), bool):
            return bool(raw.get("is_town"))
        if isinstance(raw.get("town"), bool):
            return bool(raw.get("town"))
        map_type = raw.get("map_type")
        if isinstance(map_type, str):
            lowered = map_type.lower()
            if "town" in lowered or "city" in lowered:
                return True
            if "field" in lowered or "dungeon" in lowered:
                return False
        _ = map_name
        return None

    def _coerce_float(self, value: object) -> float | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(str(value))
        except (TypeError, ValueError):
            return None

    def _coerce_int(self, value: object) -> int | None:
        coerced = self._coerce_float(value)
        if coerced is None:
            return None
        return int(coerced)

    def _coerce_bool(self, value: object) -> bool | None:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "y", "on"}:
                return True
            if lowered in {"0", "false", "no", "n", "off"}:
                return False
        if isinstance(value, (int, float)):
            return bool(value)
        return None

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
