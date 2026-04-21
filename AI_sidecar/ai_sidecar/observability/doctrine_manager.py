from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from threading import RLock


@dataclass(slots=True)
class DoctrineVersionRecord:
    version: str
    policy: dict[str, object]
    canary_percentage: float
    is_active: bool
    created_at: datetime
    activated_at: datetime | None = None
    author: str = ""


class DoctrineManager:
    def __init__(self, *, max_versions: int = 200) -> None:
        self._lock = RLock()
        self._max_versions = max(20, int(max_versions))
        self._versions: list[DoctrineVersionRecord] = [
            DoctrineVersionRecord(
                version="local-default",
                policy={
                    "conflict_resolution": {
                        "prefer_higher_priority": True,
                        "prefer_lease_owner": True,
                        "on_equal": "first_writer",
                    }
                },
                canary_percentage=100.0,
                is_active=True,
                created_at=datetime.now(UTC),
                activated_at=datetime.now(UTC),
                author="sidecar",
            )
        ]

    def publish(
        self,
        *,
        version: str,
        policy: dict[str, object],
        canary_percentage: float,
        activate: bool,
        author: str = "",
    ) -> dict[str, object]:
        ver = (version or "").strip()
        if not ver:
            return {"ok": False, "message": "version_required"}

        now = datetime.now(UTC)
        canary = max(0.0, min(100.0, float(canary_percentage)))
        with self._lock:
            for item in self._versions:
                if item.version != ver:
                    continue
                item.policy = dict(policy)
                item.canary_percentage = canary
                item.author = author
                if activate:
                    for row in self._versions:
                        row.is_active = False
                    item.is_active = True
                    item.activated_at = now
                return {"ok": True, "message": "doctrine_updated", "doctrine": asdict(item)}

            row = DoctrineVersionRecord(
                version=ver,
                policy=dict(policy),
                canary_percentage=canary,
                is_active=bool(activate),
                created_at=now,
                activated_at=now if activate else None,
                author=author,
            )
            if activate:
                for item in self._versions:
                    item.is_active = False
            self._versions.append(row)
            if len(self._versions) > self._max_versions:
                self._versions = self._versions[-self._max_versions :]
            return {"ok": True, "message": "doctrine_published", "doctrine": asdict(row)}

    def rollback(self, *, target_version: str | None = None) -> dict[str, object]:
        with self._lock:
            if target_version:
                for item in self._versions:
                    if item.version == target_version:
                        for row in self._versions:
                            row.is_active = False
                        item.is_active = True
                        item.activated_at = datetime.now(UTC)
                        return {"ok": True, "message": "doctrine_rolled_back", "doctrine": asdict(item)}
                return {"ok": False, "message": "target_version_not_found", "target_version": target_version}

            active_idx = -1
            for idx, item in enumerate(self._versions):
                if item.is_active:
                    active_idx = idx
                    break
            if active_idx <= 0:
                return {"ok": False, "message": "no_previous_version"}

            self._versions[active_idx].is_active = False
            self._versions[active_idx - 1].is_active = True
            self._versions[active_idx - 1].activated_at = datetime.now(UTC)
            return {
                "ok": True,
                "message": "doctrine_rolled_back",
                "doctrine": asdict(self._versions[active_idx - 1]),
            }

    def active(self) -> dict[str, object]:
        with self._lock:
            for item in self._versions:
                if item.is_active:
                    return asdict(item)
            return asdict(self._versions[-1])

    def list_versions(self, *, limit: int = 50) -> list[dict[str, object]]:
        with self._lock:
            rows = list(self._versions)
        rows = rows[-max(1, min(int(limit), self._max_versions)) :]
        return [asdict(item) for item in reversed(rows)]

