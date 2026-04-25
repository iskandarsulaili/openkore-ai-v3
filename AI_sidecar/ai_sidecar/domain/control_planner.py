from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

from ai_sidecar.contracts.control_domain import (
    ControlArtifactIdentity,
    ControlChangeItem,
    ControlChangePlan,
    ControlOwnerScope,
)
from ai_sidecar.domain.control_policy import ControlPolicy
from ai_sidecar.domain.control_storage import ControlStorage


@dataclass(slots=True)
class ControlPlanner:
    storage: ControlStorage

    def plan_config(
        self,
        *,
        bot_id: str,
        profile: str | None,
        target_path: str,
        desired: dict[str, str],
        policy: ControlPolicy,
        owner: ControlOwnerScope,
        policy_version: str,
    ) -> ControlChangePlan:
        identity = ControlArtifactIdentity(
            bot_id=bot_id,
            profile=profile,
            artifact_type="config",
            name=target_path,
            path=target_path,
        )
        path = self.storage.resolve_control_path(bot_id=bot_id, profile=profile, target_path=target_path)
        current_text = self.storage.read_text(path)
        parsed = self.storage.parser.parse(current_text)
        changes: list[ControlChangeItem] = []
        for key, value in desired.items():
            allowed, reason = policy.allow_write(key, owner)
            if not allowed:
                changes.append(
                    ControlChangeItem(
                        key=key,
                        previous=parsed.values.get(key, ""),
                        updated=parsed.values.get(key, ""),
                        owner=owner,
                        reason=reason,
                    )
                )
                continue
            previous = parsed.values.get(key, "")
            if previous != value:
                changes.append(
                    ControlChangeItem(
                        key=key,
                        previous=previous,
                        updated=value,
                        owner=owner,
                        reason="update",
                    )
                )
        proposed_values = dict(parsed.values)
        for change in changes:
            if change.previous != change.updated and policy.allow_write(change.key, owner)[0]:
                proposed_values[change.key] = change.updated

        checksum_before = self.storage.checksum(current_text)
        checksum_after = self.storage.checksum(self.storage.parser.render(proposed_values))
        return ControlChangePlan(
            plan_id=f"ctrl-plan-{uuid4().hex[:18]}",
            bot_id=bot_id,
            profile=profile,
            artifact=identity,
            created_at=datetime.now(UTC),
            policy_version=policy_version,
            checksum_before=checksum_before,
            checksum_after=checksum_after,
            changes=changes,
            annotations={"path": str(path)},
        )

