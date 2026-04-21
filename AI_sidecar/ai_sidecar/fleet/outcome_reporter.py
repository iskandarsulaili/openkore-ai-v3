from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from threading import RLock

from ai_sidecar.config import settings
from ai_sidecar.fleet.sync_client import FleetSyncClient


@dataclass(slots=True)
class OutcomeReporter:
    client: FleetSyncClient
    _lock: RLock = field(default_factory=RLock)
    _backlog: deque[dict[str, object]] = field(default_factory=deque)

    def report(
        self,
        *,
        bot_id: str,
        event_type: str,
        priority_class: int,
        lease_owner: str,
        conflict_key: str,
        payload: dict[str, object],
    ) -> tuple[bool, bool, dict[str, object]]:
        ok, result, _ = self.client.submit_outcome(
            bot_id=bot_id,
            event_type=event_type,
            priority_class=priority_class,
            lease_owner=lease_owner,
            conflict_key=conflict_key,
            payload=payload,
        )
        if ok:
            self.flush_backlog()
            return True, False, result

        with self._lock:
            self._backlog.append(
                {
                    "bot_id": bot_id,
                    "event_type": event_type,
                    "priority_class": int(priority_class),
                    "lease_owner": lease_owner,
                    "conflict_key": conflict_key,
                    "payload": dict(payload),
                }
            )
            while len(self._backlog) > settings.fleet_outcome_backlog_limit:
                self._backlog.popleft()
        return False, True, {}

    def flush_backlog(self) -> int:
        drained = 0
        while True:
            with self._lock:
                if not self._backlog:
                    break
                item = self._backlog[0]
            ok, _, _ = self.client.submit_outcome(
                bot_id=str(item.get("bot_id") or ""),
                event_type=str(item.get("event_type") or ""),
                priority_class=int(item.get("priority_class") or 100),
                lease_owner=str(item.get("lease_owner") or ""),
                conflict_key=str(item.get("conflict_key") or ""),
                payload=dict(item.get("payload") or {}),
            )
            if not ok:
                break
            with self._lock:
                if self._backlog:
                    self._backlog.popleft()
                    drained += 1
        return drained

    def backlog_size(self) -> int:
        with self._lock:
            return len(self._backlog)

