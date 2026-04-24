from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

from ai_sidecar.config import settings
from ai_sidecar.contracts.common import ContractMeta, utc_now

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class FleetSyncClient:
    base_url: str = settings.fleet_central_base_url
    timeout_seconds: float = settings.fleet_request_timeout_seconds
    enabled: bool = settings.fleet_central_enabled

    def _url(self, path: str) -> str:
        return self.base_url.rstrip("/") + path

    def ping_blackboard(self) -> tuple[bool, dict[str, object], str]:
        if not self.enabled:
            return False, {}, "fleet_central_disabled"
        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                resp = client.get(self._url("/v1/fleet/blackboard"))
                resp.raise_for_status()
                payload = resp.json() if resp.content else {}
            if not isinstance(payload, dict):
                return False, {}, "invalid_blackboard_payload"
            return True, payload, "ok"
        except Exception as exc:
            logger.warning(
                "fleet_sync_blackboard_failed",
                extra={"event": "fleet_sync_blackboard_failed", "error": str(exc)},
            )
            return False, {}, str(exc)

    def submit_outcome(self, *, bot_id: str, event_type: str, priority_class: int, lease_owner: str, conflict_key: str, payload: dict[str, object]) -> tuple[bool, dict[str, object], str]:
        if not self.enabled:
            return False, {}, "fleet_central_disabled"
        body = {
            "bot_id": bot_id,
            "event_type": event_type,
            "priority_class": int(priority_class),
            "lease_owner": lease_owner,
            "conflict_key": conflict_key,
            "payload": dict(payload),
        }
        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                resp = client.post(self._url("/v1/fleet/events/outcome"), json=body)
                resp.raise_for_status()
                result = resp.json() if resp.content else {}
            if not isinstance(result, dict):
                return False, {}, "invalid_outcome_response"
            return True, result, "ok"
        except Exception as exc:
            logger.warning(
                "fleet_sync_outcome_failed",
                extra={"event": "fleet_sync_outcome_failed", "bot_id": bot_id, "event_type": event_type, "error": str(exc)},
            )
            return False, {}, str(exc)

    def claim(self, *, bot_id: str, claim_type: str, map_name: str | None, channel: str, objective_id: int | None, resource_type: str | None, resource_id: str | None, quantity: int, ttl_seconds: int, priority: int, metadata: dict[str, object]) -> tuple[bool, dict[str, object], str]:
        if not self.enabled:
            return False, {}, "fleet_central_disabled"
        body = {
            "bot_id": bot_id,
            "claim_type": claim_type,
            "map_name": map_name,
            "channel": channel,
            "objective_id": objective_id,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "quantity": int(quantity),
            "ttl_seconds": int(ttl_seconds),
            "priority": int(priority),
            "metadata": dict(metadata),
        }
        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                resp = client.post(self._url("/v1/fleet/claims"), json=body)
                resp.raise_for_status()
                result = resp.json() if resp.content else {}
            if not isinstance(result, dict):
                return False, {}, "invalid_claim_response"
            return True, result, "ok"
        except Exception as exc:
            logger.warning(
                "fleet_sync_claim_failed",
                extra={"event": "fleet_sync_claim_failed", "bot_id": bot_id, "claim_type": claim_type, "error": str(exc)},
            )
            return False, {}, str(exc)

    def renew_role(self, *, bot_id: str, role: str, confidence: float, ttl_seconds: int, assigned_by: str, metadata: dict[str, object]) -> tuple[bool, dict[str, object], str]:
        if not self.enabled:
            return False, {}, "fleet_central_disabled"
        body = {
            "bot_id": bot_id,
            "role": role,
            "confidence": float(confidence),
            "ttl_seconds": int(ttl_seconds),
            "assigned_by": assigned_by,
            "metadata": dict(metadata),
        }
        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                resp = client.post(self._url("/v1/fleet/leases/role"), json=body)
                resp.raise_for_status()
                result = resp.json() if resp.content else {}
            if not isinstance(result, dict):
                return False, {}, "invalid_role_lease_response"
            return True, result, "ok"
        except Exception as exc:
            logger.warning(
                "fleet_sync_role_lease_failed",
                extra={"event": "fleet_sync_role_lease_failed", "bot_id": bot_id, "role": role, "error": str(exc)},
            )
            return False, {}, str(exc)

    def sync_payload(self, *, bot_id: str) -> dict[str, object]:
        return {
            "meta": ContractMeta(contract_version=settings.contract_version, source="fleet_sync", bot_id=bot_id).model_dump(mode="json"),
            "synced_at": utc_now().isoformat(),
        }

    async def renew_role(self, role_name: str, lease_id: str, *, bot_id: str) -> tuple[bool, dict[str, object], str]:
        """Renew a role lease on the central server."""
        if not self.enabled:
            return False, {}, "fleet_central_disabled"
        body = {
            "bot_id": bot_id,
            "role_name": role_name,
            "lease_id": lease_id,
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                resp = await client.post(self._url("/v1/fleet/roles/renew"), json=body)
                resp.raise_for_status()
                payload = resp.json() if resp.content else {}
            if not isinstance(payload, dict):
                return False, {}, "invalid_role_renew_response"
            return True, payload, "ok"
        except Exception as exc:
            logger.warning(
                "fleet_sync_role_renew_failed",
                extra={"event": "fleet_sync_role_renew_failed", "bot_id": bot_id, "role": role_name, "error": str(exc)},
            )
            return False, {}, str(exc)

    async def release_role(self, lease_id: str, *, bot_id: str) -> tuple[bool, dict[str, object], str]:
        """Voluntarily release a role lease."""
        if not self.enabled:
            return False, {}, "fleet_central_disabled"
        body = {
            "bot_id": bot_id,
            "lease_id": lease_id,
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                resp = await client.post(self._url("/v1/fleet/roles/release"), json=body)
                resp.raise_for_status()
                payload = resp.json() if resp.content else {}
            if not isinstance(payload, dict):
                return False, {}, "invalid_role_release_response"
            return True, payload, "ok"
        except Exception as exc:
            logger.warning(
                "fleet_sync_role_release_failed",
                extra={"event": "fleet_sync_role_release_failed", "bot_id": bot_id, "lease_id": lease_id, "error": str(exc)},
            )
            return False, {}, str(exc)

    async def report_zone_claim(
        self,
        map_name: str,
        purpose: str,
        duration_seconds: int,
        *,
        bot_id: str,
        conflict_key: str = "",
    ) -> tuple[bool, dict[str, object], str]:
        """Claim a zone on the central blackboard."""
        if not self.enabled:
            return False, {}, "fleet_central_disabled"
        body = {
            "bot_id": bot_id,
            "map_name": map_name,
            "purpose": purpose,
            "duration_seconds": int(duration_seconds),
            "conflict_key": conflict_key,
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                resp = await client.post(self._url("/v1/fleet/zones/claim"), json=body)
                resp.raise_for_status()
                payload = resp.json() if resp.content else {}
            if not isinstance(payload, dict):
                return False, {}, "invalid_zone_claim_response"
            return True, payload, "ok"
        except Exception as exc:
            logger.warning(
                "fleet_sync_zone_claim_failed",
                extra={"event": "fleet_sync_zone_claim_failed", "bot_id": bot_id, "map_name": map_name, "error": str(exc)},
            )
            return False, {}, str(exc)

    async def report_objective_completion(
        self,
        objective_id: str,
        result: str,
        *,
        bot_id: str,
    ) -> tuple[bool, dict[str, object], str]:
        """Mark an objective as completed."""
        if not self.enabled:
            return False, {}, "fleet_central_disabled"
        body = {"bot_id": bot_id, "result": result}
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                resp = await client.post(self._url(f"/v1/fleet/objectives/{objective_id}/complete"), json=body)
                resp.raise_for_status()
                payload = resp.json() if resp.content else {}
            if not isinstance(payload, dict):
                return False, {}, "invalid_objective_completion_response"
            return True, payload, "ok"
        except Exception as exc:
            logger.warning(
                "fleet_sync_objective_complete_failed",
                extra={"event": "fleet_sync_objective_complete_failed", "bot_id": bot_id, "objective_id": objective_id, "error": str(exc)},
            )
            return False, {}, str(exc)
