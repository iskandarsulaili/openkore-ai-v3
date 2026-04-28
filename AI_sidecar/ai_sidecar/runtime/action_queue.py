from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from threading import RLock

from ai_sidecar.contracts.actions import ActionPriorityTier, ActionProposal, ActionStatus


@dataclass(slots=True)
class QueuedAction:
    proposal: ActionProposal
    status: ActionStatus
    enqueue_seq: int
    dispatched_at: datetime | None = None
    acknowledged_at: datetime | None = None
    ack_message: str = ""


class ActionQueue:
    _PRIORITY_ORDER: dict[ActionPriorityTier, int] = {
        ActionPriorityTier.reflex: 0,
        ActionPriorityTier.tactical: 1,
        ActionPriorityTier.strategic: 2,
        ActionPriorityTier.macro_management: 3,
    }

    def __init__(self, max_per_bot: int) -> None:
        self._max_per_bot = max_per_bot
        self._lock = RLock()
        self._by_bot: dict[str, deque[QueuedAction]] = {}
        self._idempotency_index: dict[str, dict[str, str]] = {}
        self._actions_by_id: dict[str, QueuedAction] = {}
        self._action_to_bot: dict[str, str] = {}
        self._enqueue_seq = 0

    def enqueue(self, bot_id: str, proposal: ActionProposal) -> tuple[bool, ActionStatus, str, str]:
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
                    return False, existing.status, existing_action_id, "idempotent_duplicate"

            if self._normalize_datetime(proposal.expires_at) <= now:
                expired = QueuedAction(
                    proposal=proposal,
                    status=ActionStatus.expired,
                    enqueue_seq=self._next_enqueue_seq(),
                )
                self._actions_by_id[proposal.action_id] = expired
                self._action_to_bot[proposal.action_id] = bot_id
                return False, ActionStatus.expired, proposal.action_id, "action_already_expired"

            superseded_ids: list[str] = []
            if proposal.conflict_key:
                proposed_order = self._ordering_key(proposal, enqueue_seq=self._peek_next_enqueue_seq())
                for queued in bot_queue:
                    if queued.status != ActionStatus.queued:
                        continue
                    if queued.proposal.conflict_key != proposal.conflict_key:
                        continue

                    existing_order = self._ordering_key(queued.proposal, enqueue_seq=queued.enqueue_seq)
                    if proposed_order < existing_order:
                        superseded_ids.append(queued.proposal.action_id)
                        continue
                    return False, ActionStatus.superseded, queued.proposal.action_id, "conflict_key_blocked"

            if superseded_ids:
                kept: deque[QueuedAction] = deque()
                for queued in bot_queue:
                    if queued.proposal.action_id in superseded_ids:
                        superseded = replace(queued, status=ActionStatus.superseded)
                        self._actions_by_id[superseded.proposal.action_id] = superseded
                        self._clear_idempotency_index(bot_id, superseded.proposal.idempotency_key, superseded.proposal.action_id)
                        continue
                    kept.append(queued)
                bot_queue = kept
                self._by_bot[bot_id] = bot_queue

            if len(bot_queue) >= self._max_per_bot:
                dropped_for_capacity = self._select_capacity_drop_candidate(bot_queue)
                if dropped_for_capacity is None:
                    dropped_new = QueuedAction(
                        proposal=proposal,
                        status=ActionStatus.dropped,
                        enqueue_seq=self._next_enqueue_seq(),
                    )
                    self._actions_by_id[proposal.action_id] = dropped_new
                    self._action_to_bot[proposal.action_id] = bot_id
                    return False, ActionStatus.dropped, proposal.action_id, "queue_full"

                proposed_order = self._ordering_key(proposal, enqueue_seq=self._peek_next_enqueue_seq())
                existing_order = self._ordering_key(
                    dropped_for_capacity.proposal,
                    enqueue_seq=dropped_for_capacity.enqueue_seq,
                )
                if proposed_order >= existing_order:
                    dropped_new = QueuedAction(
                        proposal=proposal,
                        status=ActionStatus.dropped,
                        enqueue_seq=self._next_enqueue_seq(),
                    )
                    self._actions_by_id[proposal.action_id] = dropped_new
                    self._action_to_bot[proposal.action_id] = bot_id
                    return False, ActionStatus.dropped, proposal.action_id, "queue_full_lower_priority"

                kept: deque[QueuedAction] = deque()
                for queued in bot_queue:
                    if queued.proposal.action_id == dropped_for_capacity.proposal.action_id:
                        dropped_existing = replace(queued, status=ActionStatus.dropped)
                        self._actions_by_id[dropped_existing.proposal.action_id] = dropped_existing
                        self._clear_idempotency_index(
                            bot_id,
                            dropped_existing.proposal.idempotency_key,
                            dropped_existing.proposal.action_id,
                        )
                        continue
                    kept.append(queued)
                bot_queue = kept
                self._by_bot[bot_id] = bot_queue

            queued = QueuedAction(
                proposal=proposal,
                status=ActionStatus.queued,
                enqueue_seq=self._next_enqueue_seq(),
            )
            bot_queue.append(queued)
            idempotency[proposal.idempotency_key] = proposal.action_id
            self._actions_by_id[proposal.action_id] = queued
            self._action_to_bot[proposal.action_id] = bot_id
            return True, queued.status, queued.proposal.action_id, "action_queued"

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

            selected_idx: int | None = None
            selected_order: tuple[int, datetime, int, str] | None = None
            for idx, queued in enumerate(bot_queue):
                if queued.status != ActionStatus.queued:
                    continue
                conflict_key = queued.proposal.conflict_key
                if conflict_key and conflict_key in active_conflict_keys:
                    continue

                current_order = self._ordering_key(queued.proposal, enqueue_seq=queued.enqueue_seq)
                if selected_order is None or current_order < selected_order:
                    selected_idx = idx
                    selected_order = current_order

            if selected_idx is None:
                return None

            queued = bot_queue[selected_idx]
            queued.status = ActionStatus.dispatched
            queued.dispatched_at = now
            bot_queue[selected_idx] = queued
            self._actions_by_id[queued.proposal.action_id] = queued
            return queued.proposal

    def rollback_dispatched(self, action_id: str) -> bool:
        with self._lock:
            queued = self._actions_by_id.get(action_id)
            if queued is None or queued.status != ActionStatus.dispatched:
                return False

            bot_id = self._action_to_bot.get(action_id)
            if not bot_id:
                return False
            bot_queue = self._by_bot.get(bot_id)
            if bot_queue is None:
                return False

            for idx, item in enumerate(bot_queue):
                if item.proposal.action_id != action_id:
                    continue
                item.status = ActionStatus.queued
                item.dispatched_at = None
                bot_queue[idx] = item
                self._actions_by_id[action_id] = item
                return True

            return False

    def acknowledge(self, action_id: str, success: bool, message: str) -> tuple[bool, ActionStatus]:
        now = datetime.now(UTC)
        with self._lock:
            queued = self._actions_by_id.get(action_id)
            if queued is None:
                return False, ActionStatus.dropped
            bot_id = self._action_to_bot.get(action_id)

            if queued.status not in {ActionStatus.dispatched, ActionStatus.queued}:
                return False, queued.status

            queued.acknowledged_at = now
            queued.ack_message = message
            queued.status = ActionStatus.acknowledged if success else ActionStatus.dropped
            self._actions_by_id[action_id] = queued
            if bot_id and bot_id in self._by_bot:
                self._by_bot[bot_id] = deque(
                    item for item in self._by_bot[bot_id] if item.proposal.action_id != action_id
                )
                self._clear_idempotency_index(bot_id, queued.proposal.idempotency_key, action_id)
            return True, queued.status

    def count(self, bot_id: str) -> int:
        now = datetime.now(UTC)
        with self._lock:
            self._expire_for_bot(bot_id, now)
            return len(self._by_bot.get(bot_id, ()))

    def snapshot(self) -> dict[str, list[QueuedAction]]:
        with self._lock:
            return {bot_id: list(items) for bot_id, items in self._by_bot.items()}

    def rehydrate(self, bot_id: str, queued_actions: list[QueuedAction]) -> int:
        now = datetime.now(UTC)
        with self._lock:
            previous = self._by_bot.get(bot_id)
            if previous is not None:
                for item in previous:
                    action_id = item.proposal.action_id
                    self._actions_by_id.pop(action_id, None)
                    self._action_to_bot.pop(action_id, None)

            bot_queue: deque[QueuedAction] = deque()
            self._by_bot[bot_id] = bot_queue

            bucket = self._idempotency_index.setdefault(bot_id, {})
            bucket.clear()

            restored = 0
            for item in queued_actions:
                proposal = item.proposal
                if self._normalize_datetime(proposal.expires_at) <= now and item.status in {
                    ActionStatus.queued,
                    ActionStatus.dispatched,
                }:
                    expired = QueuedAction(
                        proposal=proposal,
                        status=ActionStatus.expired,
                        enqueue_seq=self._next_enqueue_seq(),
                        dispatched_at=item.dispatched_at,
                        acknowledged_at=item.acknowledged_at,
                        ack_message=item.ack_message,
                    )
                    self._actions_by_id[proposal.action_id] = expired
                    self._action_to_bot[proposal.action_id] = bot_id
                    continue

                restored_status = ActionStatus.queued if item.status == ActionStatus.dispatched else item.status
                restored_dispatched_at = item.dispatched_at if restored_status == ActionStatus.dispatched else None

                restored_item = QueuedAction(
                    proposal=proposal,
                    status=restored_status,
                    enqueue_seq=self._next_enqueue_seq(),
                    dispatched_at=restored_dispatched_at,
                    acknowledged_at=item.acknowledged_at,
                    ack_message=item.ack_message,
                )
                self._actions_by_id[proposal.action_id] = restored_item
                self._action_to_bot[proposal.action_id] = bot_id

                if restored_item.status in {ActionStatus.queued, ActionStatus.dispatched}:
                    bot_queue.append(restored_item)
                    bucket[proposal.idempotency_key] = proposal.action_id
                    restored += 1

            return restored

    def _expire_for_bot(self, bot_id: str, now: datetime) -> None:
        queue = self._by_bot.get(bot_id)
        if not queue:
            return
        kept: deque[QueuedAction] = deque()
        for queued in queue:
            if self._normalize_datetime(queued.proposal.expires_at) < now and queued.status in {
                ActionStatus.queued,
                ActionStatus.dispatched,
            }:
                expired = replace(queued, status=ActionStatus.expired)
                self._actions_by_id[expired.proposal.action_id] = expired
                self._clear_idempotency_index(bot_id, queued.proposal.idempotency_key, queued.proposal.action_id)
                continue
            kept.append(queued)
        self._by_bot[bot_id] = kept

    def _ordering_key(self, proposal: ActionProposal, enqueue_seq: int) -> tuple[int, datetime, int, str]:
        return (
            self._PRIORITY_ORDER.get(proposal.priority_tier, 99),
            self._normalize_datetime(proposal.created_at),
            enqueue_seq,
            proposal.action_id,
        )

    def _select_capacity_drop_candidate(self, queue: deque[QueuedAction]) -> QueuedAction | None:
        candidate: QueuedAction | None = None
        candidate_order: tuple[int, datetime, int, str] | None = None
        for queued in queue:
            if queued.status != ActionStatus.queued:
                continue
            current_order = self._ordering_key(queued.proposal, enqueue_seq=queued.enqueue_seq)
            if candidate_order is None or current_order > candidate_order:
                candidate = queued
                candidate_order = current_order
        return candidate

    def _clear_idempotency_index(self, bot_id: str, idempotency_key: str, action_id: str) -> None:
        bucket = self._idempotency_index.get(bot_id)
        if not bucket:
            return
        existing_action_id = bucket.get(idempotency_key)
        if existing_action_id == action_id:
            del bucket[idempotency_key]

    def _next_enqueue_seq(self) -> int:
        self._enqueue_seq += 1
        return self._enqueue_seq

    def _peek_next_enqueue_seq(self) -> int:
        return self._enqueue_seq + 1

    def _normalize_datetime(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
