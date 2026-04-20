from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from threading import RLock

from ai_sidecar.contracts.actions import ActionProposal, ActionStatus


@dataclass(slots=True)
class QueuedAction:
    proposal: ActionProposal
    status: ActionStatus
    dispatched_at: datetime | None = None
    acknowledged_at: datetime | None = None
    ack_message: str = ""


class ActionQueue:
    def __init__(self, max_per_bot: int) -> None:
        self._max_per_bot = max_per_bot
        self._lock = RLock()
        self._by_bot: dict[str, deque[QueuedAction]] = {}
        self._idempotency_index: dict[str, dict[str, str]] = {}
        self._actions_by_id: dict[str, QueuedAction] = {}
        self._action_to_bot: dict[str, str] = {}

    def enqueue(self, bot_id: str, proposal: ActionProposal) -> tuple[bool, ActionStatus, str]:
        now = datetime.now(UTC)
        with self._lock:
            self._expire_for_bot(bot_id, now)

            bot_queue = self._by_bot.setdefault(bot_id, deque())
            idempotency = self._idempotency_index.setdefault(bot_id, {})

            existing_action_id = idempotency.get(proposal.idempotency_key)
            if existing_action_id:
                existing = self._actions_by_id.get(existing_action_id)
                if existing and existing.status in {
                    ActionStatus.queued,
                    ActionStatus.dispatched,
                    ActionStatus.acknowledged,
                }:
                    return False, existing.status, existing_action_id

            if len(bot_queue) >= self._max_per_bot:
                dropped = bot_queue.popleft()
                dropped.status = ActionStatus.dropped
                self._actions_by_id[dropped.proposal.action_id] = dropped
                self._action_to_bot[dropped.proposal.action_id] = bot_id

            queued = QueuedAction(proposal=proposal, status=ActionStatus.queued)
            bot_queue.append(queued)
            idempotency[proposal.idempotency_key] = proposal.action_id
            self._actions_by_id[proposal.action_id] = queued
            self._action_to_bot[proposal.action_id] = bot_id
            return True, queued.status, queued.proposal.action_id

    def fetch_next(self, bot_id: str) -> ActionProposal | None:
        now = datetime.now(UTC)
        with self._lock:
            self._expire_for_bot(bot_id, now)
            bot_queue = self._by_bot.get(bot_id)
            if not bot_queue:
                return None

            active_conflict_keys: set[str] = set()
            for queued in bot_queue:
                if queued.status == ActionStatus.dispatched and queued.proposal.conflict_key:
                    active_conflict_keys.add(queued.proposal.conflict_key)

            for idx, queued in enumerate(bot_queue):
                if queued.status != ActionStatus.queued:
                    continue
                conflict_key = queued.proposal.conflict_key
                if conflict_key and conflict_key in active_conflict_keys:
                    continue

                queued.status = ActionStatus.dispatched
                queued.dispatched_at = now
                bot_queue[idx] = queued
                self._actions_by_id[queued.proposal.action_id] = queued
                return queued.proposal

            return None

    def acknowledge(self, action_id: str, success: bool, message: str) -> tuple[bool, ActionStatus]:
        now = datetime.now(UTC)
        with self._lock:
            queued = self._actions_by_id.get(action_id)
            if queued is None:
                return False, ActionStatus.dropped
            bot_id = self._action_to_bot.get(action_id)

            queued.acknowledged_at = now
            queued.ack_message = message
            queued.status = ActionStatus.acknowledged if success else ActionStatus.dropped
            self._actions_by_id[action_id] = queued
            if bot_id and bot_id in self._by_bot:
                self._by_bot[bot_id] = deque(
                    item for item in self._by_bot[bot_id] if item.proposal.action_id != action_id
                )
            return True, queued.status

    def count(self, bot_id: str) -> int:
        now = datetime.now(UTC)
        with self._lock:
            self._expire_for_bot(bot_id, now)
            return len(self._by_bot.get(bot_id, ()))

    def _expire_for_bot(self, bot_id: str, now: datetime) -> None:
        queue = self._by_bot.get(bot_id)
        if not queue:
            return
        kept: deque[QueuedAction] = deque()
        for queued in queue:
            if queued.proposal.expires_at < now and queued.status in {
                ActionStatus.queued,
                ActionStatus.dispatched,
            }:
                expired = replace(queued, status=ActionStatus.expired)
                self._actions_by_id[expired.proposal.action_id] = expired
                continue
            kept.append(queued)
        self._by_bot[bot_id] = kept
