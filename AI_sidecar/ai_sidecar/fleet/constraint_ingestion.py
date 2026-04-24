from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from threading import RLock
from typing import Any

import structlog

from ai_sidecar.config import settings

from .blackboard_types import DoctrineDirective, FleetObjective, TaskLease, ThreatBulletin, ZoneClaim

logger = structlog.get_logger(__name__)


@dataclass(slots=True)
class ConstraintIngestionState:
    _lock: RLock = field(default_factory=RLock)
    _last_sync_at: datetime | None = None
    _central_available: bool = False
    _doctrine_version: str = "local"
    _constraints_by_bot: dict[str, dict[str, object]] = field(default_factory=dict)
    _blackboard: dict[str, object] = field(default_factory=dict)
    _last_error: str = ""
    _objectives: list[FleetObjective] = field(default_factory=list)
    _zone_claims: list[ZoneClaim] = field(default_factory=list)
    _task_leases: list[TaskLease] = field(default_factory=list)
    _threats: list[ThreatBulletin] = field(default_factory=list)
    _doctrine: DoctrineDirective | None = None

    def update_from_blackboard(self, *, blackboard: dict[str, object]) -> None:
        now = datetime.now(UTC)
        constraints = blackboard.get("constraints") if isinstance(blackboard.get("constraints"), dict) else {}
        doctrine = blackboard.get("doctrine") if isinstance(blackboard.get("doctrine"), dict) else {}
        doctrine_version = str(doctrine.get("version") or "unknown")
        with self._lock:
            self._last_sync_at = now
            self._central_available = True
            self._doctrine_version = doctrine_version
            self._constraints_by_bot = {str(k): dict(v) for k, v in constraints.items() if isinstance(v, dict)}
            self._blackboard = dict(blackboard)
            self._last_error = ""
        self.parse_blackboard(blackboard)

    def mark_unavailable(self, *, reason: str) -> None:
        with self._lock:
            self._central_available = False
            self._last_error = reason

    def status(self) -> dict[str, object]:
        with self._lock:
            now = datetime.now(UTC)
            stale = True
            if self._last_sync_at is not None:
                stale = now - self._last_sync_at > timedelta(seconds=settings.fleet_local_partition_ttl_seconds)
            mode = "central" if self._central_available and not stale else "local"
            return {
                "mode": mode,
                "central_available": self._central_available,
                "stale": stale,
                "last_sync_at": self._last_sync_at,
                "doctrine_version": self._doctrine_version,
                "last_error": self._last_error,
            }

    def constraints_for_bot(self, *, bot_id: str) -> dict[str, object]:
        with self._lock:
            default = {"avoid": [], "required": [], "sources": ["local_default"]}
            row = self._constraints_by_bot.get(bot_id)
            if row is None:
                return default
            merged = dict(default)
            merged.update(row)
            return merged

    def blackboard(self) -> dict[str, object]:
        with self._lock:
            return dict(self._blackboard)

    def parse_blackboard(self, blackboard: dict[str, object]) -> None:
        """Parse raw blackboard dict into typed records."""
        now = datetime.now(UTC)
        objectives_raw = blackboard.get("objectives") if isinstance(blackboard.get("objectives"), list) else []
        zone_claims_raw = blackboard.get("zone_claims") if isinstance(blackboard.get("zone_claims"), list) else []
        task_leases_raw = blackboard.get("task_leases") if isinstance(blackboard.get("task_leases"), list) else []
        threats_raw = blackboard.get("threats") if isinstance(blackboard.get("threats"), list) else []
        if not threats_raw:
            threats_raw = blackboard.get("bulletins") if isinstance(blackboard.get("bulletins"), list) else []
        doctrine_raw = blackboard.get("doctrine") if isinstance(blackboard.get("doctrine"), dict) else {}

        objectives: list[FleetObjective] = []
        for item in objectives_raw:
            if not isinstance(item, dict):
                continue
            payload = self._normalize_objective(item, now=now)
            if payload is None:
                continue
            try:
                objectives.append(FleetObjective.model_validate(payload))
            except Exception as exc:
                logger.warning("fleet_blackboard_objective_parse_failed", error=str(exc), payload=payload)

        zone_claims: list[ZoneClaim] = []
        for item in zone_claims_raw:
            if not isinstance(item, dict):
                continue
            payload = self._normalize_zone_claim(item, now=now)
            if payload is None:
                continue
            try:
                zone_claims.append(ZoneClaim.model_validate(payload))
            except Exception as exc:
                logger.warning("fleet_blackboard_zone_claim_parse_failed", error=str(exc), payload=payload)

        task_leases: list[TaskLease] = []
        for item in task_leases_raw:
            if not isinstance(item, dict):
                continue
            payload = self._normalize_task_lease(item, now=now)
            if payload is None:
                continue
            try:
                task_leases.append(TaskLease.model_validate(payload))
            except Exception as exc:
                logger.warning("fleet_blackboard_task_lease_parse_failed", error=str(exc), payload=payload)

        threats: list[ThreatBulletin] = []
        for item in threats_raw:
            if not isinstance(item, dict):
                continue
            payload = self._normalize_threat(item, now=now)
            if payload is None:
                continue
            try:
                threats.append(ThreatBulletin.model_validate(payload))
            except Exception as exc:
                logger.warning("fleet_blackboard_threat_parse_failed", error=str(exc), payload=payload)

        doctrine: DoctrineDirective | None = None
        if doctrine_raw:
            payload = self._normalize_doctrine(doctrine_raw, now=now)
            if payload is not None:
                try:
                    doctrine = DoctrineDirective.model_validate(payload)
                except Exception as exc:
                    logger.warning("fleet_blackboard_doctrine_parse_failed", error=str(exc), payload=payload)

        with self._lock:
            self._objectives = objectives
            self._zone_claims = zone_claims
            self._task_leases = task_leases
            self._threats = threats
            self._doctrine = doctrine

    def get_active_objectives(self) -> list[FleetObjective]:
        with self._lock:
            return [item for item in self._objectives if item.status == "active"]

    def get_zone_claim(self, map_name: str) -> ZoneClaim | None:
        with self._lock:
            now = datetime.now(UTC)
            for claim in self._zone_claims:
                expires_at = claim.expires_at
                if expires_at.tzinfo is None:
                    expires_at = expires_at.replace(tzinfo=UTC)
                if claim.map_name == map_name and expires_at > now:
                    return claim
        return None

    def is_zone_claimed(self, map_name: str, bot_id: str) -> bool:
        claim = self.get_zone_claim(map_name)
        return claim is not None and claim.claimed_by != bot_id

    def get_task_lease(self, lease_id: str) -> TaskLease | None:
        with self._lock:
            for lease in self._task_leases:
                if lease.lease_id == lease_id:
                    return lease
        return None

    def get_active_threats(self, map_name: str | None = None) -> list[ThreatBulletin]:
        now = datetime.now(UTC)
        with self._lock:
            threats = [item for item in self._threats if self._normalize_dt(item.expires_at, fallback=now) > now]
            if map_name:
                threats = [item for item in threats if item.map_name == map_name]
            return threats

    def get_doctrine(self) -> DoctrineDirective | None:
        with self._lock:
            if self._doctrine and self._doctrine.active:
                return self._doctrine
            return None

    def _normalize_objective(self, item: dict[str, object], *, now: datetime) -> dict[str, object] | None:
        objective_id = str(item.get("objective_id") or item.get("id") or item.get("title") or "").strip()
        if not objective_id:
            return None
        objective_type = str(item.get("objective_type") or item.get("type") or item.get("category") or "unknown")
        target_map = item.get("target_map") or item.get("map_name") or item.get("map")
        target_mob = item.get("target_mob") or item.get("mob")
        target_item = item.get("target_item") or item.get("item")
        assigned = item.get("assigned_bots") if isinstance(item.get("assigned_bots"), list) else item.get("assigned")
        assigned_bots = [str(bot) for bot in list(assigned or []) if str(bot).strip()]
        start_time = self._normalize_dt(item.get("start_time") or item.get("created_at"), fallback=now)
        deadline = self._normalize_dt(item.get("deadline") or item.get("expires_at"), fallback=start_time)
        return {
            "objective_id": objective_id,
            "objective_type": objective_type,
            "target_map": (str(target_map) if target_map is not None else None),
            "target_mob": (str(target_mob) if target_mob is not None else None),
            "target_item": (str(target_item) if target_item is not None else None),
            "priority": int(item.get("priority") or 0),
            "assigned_bots": assigned_bots,
            "min_bots": int(item.get("min_bots") or item.get("min_assignees") or 1),
            "max_bots": int(item.get("max_bots") or item.get("max_assignees") or max(1, len(assigned_bots) or 1)),
            "start_time": start_time,
            "deadline": deadline,
            "status": str(item.get("status") or "active"),
        }

    def _normalize_zone_claim(self, item: dict[str, object], *, now: datetime) -> dict[str, object] | None:
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        map_name = str(item.get("map_name") or "").strip()
        if not map_name:
            return None
        channel = str(item.get("channel") or "")
        conflict_key = str(
            item.get("conflict_key")
            or (metadata.get("conflict_key") if isinstance(metadata, dict) else "")
            or f"{map_name}:{channel}"
        )
        claimed_at = self._normalize_dt(item.get("claimed_at") or item.get("created_at"), fallback=now)
        expires_at = self._normalize_dt(item.get("expires_at"), fallback=now)
        return {
            "zone_id": str(item.get("zone_id") or item.get("claim_id") or item.get("id") or conflict_key or map_name),
            "map_name": map_name,
            "claimed_by": str(item.get("claimed_by") or item.get("bot_id") or ""),
            "claimed_at": claimed_at,
            "expires_at": expires_at,
            "purpose": str(item.get("purpose") or item.get("claim_type") or metadata.get("purpose") or ""),
            "conflict_key": conflict_key,
        }

    def _normalize_task_lease(self, item: dict[str, object], *, now: datetime) -> dict[str, object] | None:
        lease_id = str(item.get("lease_id") or item.get("id") or "").strip()
        if not lease_id:
            return None
        payload = item.get("payload") if isinstance(item.get("payload"), dict) else {}
        params = item.get("params") if isinstance(item.get("params"), dict) else payload
        issued_at = self._normalize_dt(item.get("issued_at") or item.get("created_at"), fallback=now)
        deadline = self._normalize_dt(item.get("deadline") or item.get("expires_at"), fallback=issued_at)
        return {
            "lease_id": lease_id,
            "task_type": str(item.get("task_type") or item.get("lease_type") or payload.get("task_type") or "task"),
            "assigned_to": str(item.get("assigned_to") or item.get("bot_id") or payload.get("bot_id") or ""),
            "params": dict(params or {}),
            "issued_at": issued_at,
            "deadline": deadline,
            "status": str(item.get("status") or "active"),
        }

    def _normalize_threat(self, item: dict[str, object], *, now: datetime) -> dict[str, object] | None:
        threat_id = str(item.get("threat_id") or item.get("bulletin_id") or item.get("id") or "").strip()
        if not threat_id:
            return None
        context = item.get("context") if isinstance(item.get("context"), dict) else {}
        location = item.get("location") or context.get("location")
        map_name = str(item.get("map_name") or context.get("map") or context.get("map_name") or location or "unknown")
        coords = item.get("coordinates") or context.get("coordinates") or context.get("coords")
        coordinates = None
        if isinstance(coords, (list, tuple)) and len(coords) == 2:
            try:
                coordinates = (int(coords[0]), int(coords[1]))
            except Exception:
                coordinates = None
        severity = self._normalize_severity(item.get("severity") or context.get("severity"))
        reported_at = self._normalize_dt(item.get("reported_at") or item.get("created_at"), fallback=now)
        ttl_seconds = context.get("ttl_seconds") if isinstance(context.get("ttl_seconds"), (int, float)) else None
        fallback_expiry = reported_at + timedelta(seconds=int(ttl_seconds or 3600))
        expires_at = self._normalize_dt(item.get("expires_at") or context.get("expires_at"), fallback=fallback_expiry)
        return {
            "threat_id": threat_id,
            "threat_type": str(item.get("threat_type") or context.get("threat_type") or "unknown"),
            "map_name": map_name,
            "coordinates": coordinates,
            "severity": severity,
            "reported_by": str(item.get("reported_by") or context.get("reported_by") or "central"),
            "reported_at": reported_at,
            "expires_at": expires_at,
        }

    def _normalize_doctrine(self, doctrine: dict[str, object], *, now: datetime) -> dict[str, object] | None:
        raw_version = doctrine.get("version") or doctrine.get("doctrine_version")
        try:
            version = int(raw_version)
        except Exception:
            version = 0
        issued_at = self._normalize_dt(doctrine.get("issued_at") or doctrine.get("effective_from"), fallback=now)
        rules: list[str] = []
        raw_rules = doctrine.get("rules")
        if isinstance(raw_rules, list):
            rules = [str(rule) for rule in raw_rules if str(rule).strip()]
        policy = doctrine.get("policy") if isinstance(doctrine.get("policy"), dict) else {}
        if isinstance(policy, dict):
            rule_rows = policy.get("rules")
            if isinstance(rule_rows, list):
                for rule in rule_rows:
                    text = str(rule).strip()
                    if text:
                        rules.append(text)
            avoid = policy.get("avoid_map") or policy.get("avoid_maps") or policy.get("avoid")
            if isinstance(avoid, str) and avoid.strip():
                rules.append(f"avoid_map:{avoid.strip()}")
            elif isinstance(avoid, list):
                for map_name in avoid:
                    text = str(map_name).strip()
                    if text:
                        rules.append(f"avoid_map:{text}")
        return {
            "doctrine_id": str(doctrine.get("doctrine_id") or doctrine.get("id") or raw_version or "local"),
            "version": version,
            "issued_at": issued_at,
            "rules": rules,
            "active": bool(doctrine.get("active", True)),
        }

    def _normalize_dt(self, value: object, *, fallback: datetime) -> datetime:
        if isinstance(value, datetime):
            return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
        if isinstance(value, str) and value:
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)
            except Exception:
                return fallback
        return fallback

    def _normalize_severity(self, value: object) -> int:
        if isinstance(value, (int, float)):
            return max(1, int(value))
        normalized = str(value or "").strip().lower()
        mapping = {
            "info": 1,
            "low": 1,
            "medium": 2,
            "warning": 3,
            "high": 4,
            "critical": 5,
        }
        return mapping.get(normalized, 1)
