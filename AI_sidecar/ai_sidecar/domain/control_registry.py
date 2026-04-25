from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import RLock

from ai_sidecar.contracts.control_domain import ControlArtifactIdentity, ControlArtifactRecord, ControlOwnerScope


@dataclass(slots=True)
class ControlRegistry:
    _lock: RLock = field(default_factory=RLock)
    _artifacts: dict[str, ControlArtifactRecord] = field(default_factory=dict)

    def upsert(
        self,
        *,
        identity: ControlArtifactIdentity,
        owner: ControlOwnerScope,
        checksum: str,
        version: str,
        metadata: dict[str, object] | None = None,
        updated_at: datetime | None = None,
    ) -> ControlArtifactRecord:
        record = ControlArtifactRecord(
            identity=identity,
            owner=owner,
            checksum=checksum,
            version=version,
            updated_at=updated_at or datetime.now(UTC),
            metadata=dict(metadata or {}),
        )
        with self._lock:
            self._artifacts[self._key(identity)] = record
        return record

    def get(self, identity: ControlArtifactIdentity) -> ControlArtifactRecord | None:
        with self._lock:
            return self._artifacts.get(self._key(identity))

    def list_for_bot(self, *, bot_id: str) -> list[ControlArtifactRecord]:
        with self._lock:
            records = [item for item in self._artifacts.values() if item.identity.bot_id == bot_id]
        records.sort(key=lambda rec: (rec.identity.artifact_type.value, rec.identity.name))
        return records

    def _key(self, identity: ControlArtifactIdentity) -> str:
        profile = identity.profile or ""
        return f"{identity.bot_id}|{profile}|{identity.artifact_type.value}|{identity.name}"

