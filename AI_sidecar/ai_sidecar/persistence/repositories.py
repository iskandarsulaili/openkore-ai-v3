from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from ai_sidecar.contracts.actions import ActionProposal, ActionStatus
from ai_sidecar.contracts.state import BotStateSnapshot
from ai_sidecar.contracts.telemetry import TelemetryEvent
from ai_sidecar.persistence.db import SQLiteDB
from ai_sidecar.persistence.models import (
    ActionRecord,
    AuditEventRecord,
    BotIdentityRecord,
    MacroPublicationRecord,
    MemoryEpisodeRecord,
    MemorySemanticRecord,
    SnapshotRecord,
    TelemetryEventRecord,
)

logger = logging.getLogger(__name__)


def utc_now() -> datetime:
    return datetime.now(UTC)


def to_iso(value: datetime) -> str:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC).isoformat()
    return value.astimezone(UTC).isoformat()


def from_iso(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def to_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def from_json(value: str) -> object:
    return json.loads(value)


class BotRepository:
    def __init__(self, db: SQLiteDB) -> None:
        self._db = db

    def upsert_registration(
        self,
        *,
        bot_id: str,
        bot_name: str | None,
        capabilities: list[str],
        attributes: dict[str, str],
        role: str | None,
        assignment: str | None,
        tick_id: str | None = None,
        liveness_state: str = "online",
    ) -> BotIdentityRecord:
        now = utc_now()
        self._db.execute(
            """
            INSERT INTO bot_registry(
                bot_id, bot_name, role, assignment, capabilities_json, attributes_json,
                first_seen_at, last_seen_at, last_tick_id, liveness_state
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(bot_id) DO UPDATE SET
                bot_name=COALESCE(excluded.bot_name, bot_registry.bot_name),
                role=COALESCE(excluded.role, bot_registry.role),
                assignment=COALESCE(excluded.assignment, bot_registry.assignment),
                capabilities_json=excluded.capabilities_json,
                attributes_json=excluded.attributes_json,
                last_seen_at=excluded.last_seen_at,
                last_tick_id=COALESCE(excluded.last_tick_id, bot_registry.last_tick_id),
                liveness_state=excluded.liveness_state
            """,
            (
                bot_id,
                bot_name,
                role,
                assignment,
                to_json(capabilities),
                to_json(attributes),
                to_iso(now),
                to_iso(now),
                tick_id,
                liveness_state,
            ),
        )
        current = self.get(bot_id)
        if current is None:
            raise RuntimeError(f"upsert failed for bot_id={bot_id}")
        return current

    def touch(
        self,
        *,
        bot_id: str,
        tick_id: str | None = None,
        liveness_state: str = "online",
    ) -> None:
        now = utc_now()
        self._db.execute(
            """
            INSERT INTO bot_registry(
                bot_id, bot_name, role, assignment, capabilities_json, attributes_json,
                first_seen_at, last_seen_at, last_tick_id, liveness_state
            ) VALUES (?, NULL, NULL, NULL, '[]', '{}', ?, ?, ?, ?)
            ON CONFLICT(bot_id) DO UPDATE SET
                last_seen_at=excluded.last_seen_at,
                last_tick_id=COALESCE(excluded.last_tick_id, bot_registry.last_tick_id),
                liveness_state=excluded.liveness_state
            """,
            (bot_id, to_iso(now), to_iso(now), tick_id, liveness_state),
        )

    def update_assignment(
        self,
        *,
        bot_id: str,
        role: str | None,
        assignment: str | None,
        attributes: dict[str, str] | None,
    ) -> BotIdentityRecord | None:
        existing = self.get(bot_id)
        if existing is None:
            return None
        merged_attributes = dict(existing.attributes)
        if attributes:
            merged_attributes.update(attributes)
        self._db.execute(
            """
            UPDATE bot_registry
            SET role=?, assignment=?, attributes_json=?, last_seen_at=?, liveness_state='online'
            WHERE bot_id=?
            """,
            (role, assignment, to_json(merged_attributes), to_iso(utc_now()), bot_id),
        )
        return self.get(bot_id)

    def get(self, bot_id: str) -> BotIdentityRecord | None:
        row = self._db.fetchone("SELECT * FROM bot_registry WHERE bot_id = ?", (bot_id,))
        if row is None:
            return None
        return BotIdentityRecord(
            bot_id=row["bot_id"],
            bot_name=row["bot_name"],
            role=row["role"],
            assignment=row["assignment"],
            capabilities=list(from_json(row["capabilities_json"])),
            attributes=dict(from_json(row["attributes_json"])),
            first_seen_at=from_iso(row["first_seen_at"]),
            last_seen_at=from_iso(row["last_seen_at"]),
            last_tick_id=row["last_tick_id"],
            liveness_state=row["liveness_state"],
        )

    def list_all(self) -> list[BotIdentityRecord]:
        rows = self._db.fetchall("SELECT * FROM bot_registry ORDER BY last_seen_at DESC")
        return [
            BotIdentityRecord(
                bot_id=row["bot_id"],
                bot_name=row["bot_name"],
                role=row["role"],
                assignment=row["assignment"],
                capabilities=list(from_json(row["capabilities_json"])),
                attributes=dict(from_json(row["attributes_json"])),
                first_seen_at=from_iso(row["first_seen_at"]),
                last_seen_at=from_iso(row["last_seen_at"]),
                last_tick_id=row["last_tick_id"],
                liveness_state=row["liveness_state"],
            )
            for row in rows
        ]

    def count(self) -> int:
        row = self._db.fetchone("SELECT COUNT(*) AS c FROM bot_registry")
        return int(row["c"]) if row else 0


class SnapshotRepository:
    def __init__(self, db: SQLiteDB, max_history_per_bot: int) -> None:
        self._db = db
        self._max_history_per_bot = max_history_per_bot

    def save_snapshot(self, snapshot: BotStateSnapshot) -> None:
        now = utc_now()
        self._db.execute(
            """
            INSERT INTO snapshots(bot_id, tick_id, observed_at, ingested_at, snapshot_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                snapshot.meta.bot_id,
                snapshot.tick_id,
                to_iso(snapshot.observed_at),
                to_iso(now),
                to_json(snapshot.model_dump(mode="json")),
            ),
        )
        self._trim_history(snapshot.meta.bot_id)

    def latest_snapshot(self, bot_id: str) -> BotStateSnapshot | None:
        row = self._db.fetchone(
            "SELECT snapshot_json FROM snapshots WHERE bot_id=? ORDER BY observed_at DESC, id DESC LIMIT 1",
            (bot_id,),
        )
        if row is None:
            return None
        payload = from_json(row["snapshot_json"])
        return BotStateSnapshot.model_validate(payload)

    def list_recent(self, *, bot_id: str | None, limit: int) -> list[SnapshotRecord]:
        if bot_id:
            rows = self._db.fetchall(
                "SELECT * FROM snapshots WHERE bot_id=? ORDER BY observed_at DESC, id DESC LIMIT ?",
                (bot_id, limit),
            )
        else:
            rows = self._db.fetchall("SELECT * FROM snapshots ORDER BY observed_at DESC, id DESC LIMIT ?", (limit,))

        return [
            SnapshotRecord(
                id=int(row["id"]),
                bot_id=row["bot_id"],
                tick_id=row["tick_id"],
                observed_at=from_iso(row["observed_at"]),
                ingested_at=from_iso(row["ingested_at"]),
                snapshot=dict(from_json(row["snapshot_json"])),
            )
            for row in rows
        ]

    def count(self, bot_id: str | None = None) -> int:
        if bot_id:
            row = self._db.fetchone("SELECT COUNT(*) AS c FROM snapshots WHERE bot_id=?", (bot_id,))
        else:
            row = self._db.fetchone("SELECT COUNT(*) AS c FROM snapshots")
        return int(row["c"]) if row else 0

    def _trim_history(self, bot_id: str) -> None:
        self._db.execute(
            """
            DELETE FROM snapshots
            WHERE bot_id=?
              AND id NOT IN (
                SELECT id
                FROM snapshots
                WHERE bot_id=?
                ORDER BY observed_at DESC, id DESC
                LIMIT ?
              )
            """,
            (bot_id, bot_id, self._max_history_per_bot),
        )


class ActionRepository:
    def __init__(self, db: SQLiteDB) -> None:
        self._db = db

    def upsert_action(
        self,
        *,
        bot_id: str,
        proposal: ActionProposal,
        status: ActionStatus,
        status_reason: str,
    ) -> None:
        now = utc_now()
        self._db.execute(
            """
            INSERT INTO actions(
                action_id, bot_id, kind, status, priority_tier, conflict_key, idempotency_key,
                proposal_json, created_at, expires_at, queued_at, dispatched_at,
                acknowledged_at, ack_success, ack_result_code, ack_message, poll_id, status_reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, '', NULL, ?)
            ON CONFLICT(action_id) DO UPDATE SET
                status=excluded.status,
                status_reason=excluded.status_reason,
                proposal_json=excluded.proposal_json,
                expires_at=excluded.expires_at,
                kind=excluded.kind,
                priority_tier=excluded.priority_tier,
                conflict_key=excluded.conflict_key,
                idempotency_key=excluded.idempotency_key
            """,
            (
                proposal.action_id,
                bot_id,
                proposal.kind,
                status.value,
                proposal.priority_tier.value,
                proposal.conflict_key,
                proposal.idempotency_key,
                to_json(proposal.model_dump(mode="json")),
                to_iso(proposal.created_at),
                to_iso(proposal.expires_at),
                to_iso(now),
                status_reason,
            ),
        )

    def mark_dispatched(self, *, action_id: str, poll_id: str | None = None) -> None:
        self._db.execute(
            """
            UPDATE actions
            SET status=?, dispatched_at=?, poll_id=COALESCE(?, poll_id), status_reason=?
            WHERE action_id=?
            """,
            (ActionStatus.dispatched.value, to_iso(utc_now()), poll_id, "action_dispatched", action_id),
        )

    def mark_acknowledged(
        self,
        *,
        action_id: str,
        success: bool,
        result_code: str,
        message: str,
        poll_id: str | None,
    ) -> None:
        status = ActionStatus.acknowledged if success else ActionStatus.dropped
        self._db.execute(
            """
            UPDATE actions
            SET status=?,
                acknowledged_at=?,
                ack_success=?,
                ack_result_code=?,
                ack_message=?,
                poll_id=COALESCE(?, poll_id),
                status_reason=?
            WHERE action_id=?
            """,
            (
                status.value,
                to_iso(utc_now()),
                1 if success else 0,
                result_code,
                message,
                poll_id,
                "ack_success" if success else "ack_failed",
                action_id,
            ),
        )

    def list_recent(self, *, bot_id: str, limit: int) -> list[ActionRecord]:
        rows = self._db.fetchall(
            "SELECT * FROM actions WHERE bot_id=? ORDER BY queued_at DESC, action_id DESC LIMIT ?",
            (bot_id, limit),
        )
        return [self._row_to_action(row) for row in rows]

    def count(self, *, bot_id: str | None = None, status: ActionStatus | None = None) -> int:
        if bot_id and status:
            row = self._db.fetchone(
                "SELECT COUNT(*) AS c FROM actions WHERE bot_id=? AND status=?",
                (bot_id, status.value),
            )
        elif bot_id:
            row = self._db.fetchone("SELECT COUNT(*) AS c FROM actions WHERE bot_id=?", (bot_id,))
        elif status:
            row = self._db.fetchone("SELECT COUNT(*) AS c FROM actions WHERE status=?", (status.value,))
        else:
            row = self._db.fetchone("SELECT COUNT(*) AS c FROM actions")
        return int(row["c"]) if row else 0

    def _row_to_action(self, row: object) -> ActionRecord:
        return ActionRecord(
            action_id=row["action_id"],
            bot_id=row["bot_id"],
            kind=row["kind"],
            status=row["status"],
            priority_tier=row["priority_tier"],
            conflict_key=row["conflict_key"],
            idempotency_key=row["idempotency_key"],
            proposal=dict(from_json(row["proposal_json"])),
            created_at=from_iso(row["created_at"]),
            expires_at=from_iso(row["expires_at"]),
            queued_at=from_iso(row["queued_at"]),
            dispatched_at=from_iso(row["dispatched_at"]) if row["dispatched_at"] else None,
            acknowledged_at=from_iso(row["acknowledged_at"]) if row["acknowledged_at"] else None,
            ack_success=bool(row["ack_success"]) if row["ack_success"] is not None else None,
            ack_result_code=row["ack_result_code"],
            ack_message=row["ack_message"] or "",
            poll_id=row["poll_id"],
            status_reason=row["status_reason"] or "",
        )


class TelemetryRepository:
    _INCIDENT_LEVELS = {"warning", "error"}

    def __init__(self, db: SQLiteDB, max_per_bot: int, operational_window_minutes: int) -> None:
        self._db = db
        self._max_per_bot = max_per_bot
        self._operational_window = operational_window_minutes

    def ingest(self, *, bot_id: str, events: list[TelemetryEvent]) -> tuple[int, int]:
        if not events:
            return 0, 0

        now = utc_now()
        rows: list[tuple[object, ...]] = []
        for item in events:
            level = item.level.value
            rows.append(
                (
                    bot_id,
                    to_iso(item.timestamp),
                    level,
                    item.category,
                    item.event,
                    item.message,
                    to_json(item.metrics),
                    to_json(item.tags),
                    to_iso(now),
                    1 if level in self._INCIDENT_LEVELS else 0,
                )
            )

        self._db.executemany(
            """
            INSERT INTO telemetry_events(
                bot_id, timestamp, level, category, event, message,
                metrics_json, tags_json, ingested_at, is_incident
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

        dropped = self._trim_bot_telemetry(bot_id)
        return len(events), dropped

    def increment_counter(self, *, bot_id: str, name: str, delta: int = 1) -> None:
        counter_key = f"{bot_id}:{name}"
        self._db.execute(
            """
            INSERT INTO telemetry_counters(counter_key, bot_id, name, value, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(counter_key) DO UPDATE SET
                value=telemetry_counters.value + excluded.value,
                updated_at=excluded.updated_at
            """,
            (counter_key, bot_id, name, int(delta), to_iso(utc_now())),
        )

    def get_counters(self, *, bot_id: str | None = None) -> dict[str, int]:
        if bot_id:
            rows = self._db.fetchall("SELECT name, value FROM telemetry_counters WHERE bot_id=?", (bot_id,))
        else:
            rows = self._db.fetchall("SELECT name, SUM(value) AS value FROM telemetry_counters GROUP BY name")
        return {str(row["name"]): int(row["value"]) for row in rows}

    def recent_events(self, *, bot_id: str | None, limit: int, incidents_only: bool = False) -> list[TelemetryEventRecord]:
        sql = "SELECT * FROM telemetry_events"
        clauses: list[str] = []
        params: list[object] = []
        if bot_id:
            clauses.append("bot_id=?")
            params.append(bot_id)
        if incidents_only:
            clauses.append("is_incident=1")
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        rows = self._db.fetchall(sql, tuple(params))
        return [self._row_to_telemetry(row) for row in rows]

    def operational_summary(self, *, bot_id: str | None, incidents_limit: int) -> dict[str, object]:
        since = utc_now() - timedelta(minutes=self._operational_window)
        if bot_id:
            total_row = self._db.fetchone(
                "SELECT COUNT(*) AS c FROM telemetry_events WHERE bot_id=? AND timestamp>=?",
                (bot_id, to_iso(since)),
            )
            level_rows = self._db.fetchall(
                """
                SELECT level, COUNT(*) AS c
                FROM telemetry_events
                WHERE bot_id=? AND timestamp>=?
                GROUP BY level
                """,
                (bot_id, to_iso(since)),
            )
            event_rows = self._db.fetchall(
                """
                SELECT event, COUNT(*) AS c
                FROM telemetry_events
                WHERE bot_id=? AND timestamp>=?
                GROUP BY event
                ORDER BY c DESC, event ASC
                LIMIT 10
                """,
                (bot_id, to_iso(since)),
            )
        else:
            total_row = self._db.fetchone("SELECT COUNT(*) AS c FROM telemetry_events WHERE timestamp>=?", (to_iso(since),))
            level_rows = self._db.fetchall(
                "SELECT level, COUNT(*) AS c FROM telemetry_events WHERE timestamp>=? GROUP BY level",
                (to_iso(since),),
            )
            event_rows = self._db.fetchall(
                """
                SELECT event, COUNT(*) AS c
                FROM telemetry_events
                WHERE timestamp>=?
                GROUP BY event
                ORDER BY c DESC, event ASC
                LIMIT 10
                """,
                (to_iso(since),),
            )

        incidents = self.recent_events(bot_id=bot_id, limit=incidents_limit, incidents_only=True)
        return {
            "window_minutes": self._operational_window,
            "window_since": to_iso(since),
            "total_events": int(total_row["c"]) if total_row else 0,
            "levels": {str(row["level"]): int(row["c"]) for row in level_rows},
            "top_events": [{"event": str(row["event"]), "count": int(row["c"])} for row in event_rows],
            "recent_incidents": [
                {
                    "id": item.id,
                    "bot_id": item.bot_id,
                    "timestamp": item.timestamp.isoformat(),
                    "level": item.level,
                    "category": item.category,
                    "event": item.event,
                    "message": item.message,
                    "tags": item.tags,
                }
                for item in incidents
            ],
        }

    def count(self, bot_id: str | None = None) -> int:
        if bot_id:
            row = self._db.fetchone("SELECT COUNT(*) AS c FROM telemetry_events WHERE bot_id=?", (bot_id,))
        else:
            row = self._db.fetchone("SELECT COUNT(*) AS c FROM telemetry_events")
        return int(row["c"]) if row else 0

    def _trim_bot_telemetry(self, bot_id: str) -> int:
        before = self._db.fetchone("SELECT COUNT(*) AS c FROM telemetry_events WHERE bot_id=?", (bot_id,))
        count_before = int(before["c"]) if before else 0
        self._db.execute(
            """
            DELETE FROM telemetry_events
            WHERE bot_id=?
              AND id NOT IN (
                SELECT id
                FROM telemetry_events
                WHERE bot_id=?
                ORDER BY id DESC
                LIMIT ?
              )
            """,
            (bot_id, bot_id, self._max_per_bot),
        )
        after = self._db.fetchone("SELECT COUNT(*) AS c FROM telemetry_events WHERE bot_id=?", (bot_id,))
        count_after = int(after["c"]) if after else 0
        return max(0, count_before - count_after)

    def _row_to_telemetry(self, row: object) -> TelemetryEventRecord:
        return TelemetryEventRecord(
            id=int(row["id"]),
            bot_id=row["bot_id"],
            timestamp=from_iso(row["timestamp"]),
            level=row["level"],
            category=row["category"],
            event=row["event"],
            message=row["message"],
            metrics=dict(from_json(row["metrics_json"])),
            tags=dict(from_json(row["tags_json"])),
            ingested_at=from_iso(row["ingested_at"]),
            is_incident=bool(row["is_incident"]),
        )


class MacroRepository:
    def __init__(self, db: SQLiteDB) -> None:
        self._db = db

    def save_publication(
        self,
        *,
        bot_id: str,
        publication_id: str,
        version: str,
        content_sha256: str,
        published_at: datetime,
        manifest: dict[str, object],
        paths: dict[str, str],
    ) -> None:
        self._db.execute(
            """
            INSERT INTO macro_publications(
                publication_id, bot_id, version, content_sha256, published_at,
                manifest_json, paths_json, macro_count, event_macro_count, automacro_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(publication_id) DO UPDATE SET
                bot_id=excluded.bot_id,
                version=excluded.version,
                content_sha256=excluded.content_sha256,
                published_at=excluded.published_at,
                manifest_json=excluded.manifest_json,
                paths_json=excluded.paths_json,
                macro_count=excluded.macro_count,
                event_macro_count=excluded.event_macro_count,
                automacro_count=excluded.automacro_count
            """,
            (
                publication_id,
                bot_id,
                version,
                content_sha256,
                to_iso(published_at),
                to_json(manifest),
                to_json(paths),
                int(manifest.get("macro_count", 0)),
                int(manifest.get("event_macro_count", 0)),
                int(manifest.get("automacro_count", 0)),
            ),
        )

    def latest_for_bot(self, bot_id: str) -> MacroPublicationRecord | None:
        row = self._db.fetchone(
            "SELECT * FROM macro_publications WHERE bot_id=? ORDER BY published_at DESC LIMIT 1",
            (bot_id,),
        )
        if row is None:
            return None
        return self._row_to_macro(row)

    def list_recent(self, *, bot_id: str | None, limit: int) -> list[MacroPublicationRecord]:
        if bot_id:
            rows = self._db.fetchall(
                "SELECT * FROM macro_publications WHERE bot_id=? ORDER BY published_at DESC LIMIT ?",
                (bot_id, limit),
            )
        else:
            rows = self._db.fetchall("SELECT * FROM macro_publications ORDER BY published_at DESC LIMIT ?", (limit,))
        return [self._row_to_macro(row) for row in rows]

    def _row_to_macro(self, row: object) -> MacroPublicationRecord:
        return MacroPublicationRecord(
            publication_id=row["publication_id"],
            bot_id=row["bot_id"],
            version=row["version"],
            content_sha256=row["content_sha256"],
            published_at=from_iso(row["published_at"]),
            manifest=dict(from_json(row["manifest_json"])),
            paths=dict(from_json(row["paths_json"])),
            macro_count=int(row["macro_count"]),
            event_macro_count=int(row["event_macro_count"]),
            automacro_count=int(row["automacro_count"]),
        )


class AuditRepository:
    def __init__(self, db: SQLiteDB, max_history: int) -> None:
        self._db = db
        self._max_history = max_history

    def record(
        self,
        *,
        level: str,
        event_type: str,
        summary: str,
        bot_id: str | None,
        payload: dict[str, object],
    ) -> None:
        self._db.execute(
            """
            INSERT INTO audit_events(timestamp, level, event_type, bot_id, summary, payload_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (to_iso(utc_now()), level, event_type, bot_id, summary, to_json(payload)),
        )
        self._db.execute(
            """
            DELETE FROM audit_events
            WHERE id NOT IN (
                SELECT id
                FROM audit_events
                ORDER BY id DESC
                LIMIT ?
            )
            """,
            (self._max_history,),
        )

    def recent(
        self,
        *,
        limit: int,
        bot_id: str | None = None,
        event_type: str | None = None,
    ) -> list[AuditEventRecord]:
        sql = "SELECT * FROM audit_events"
        clauses: list[str] = []
        params: list[object] = []
        if bot_id:
            clauses.append("bot_id=?")
            params.append(bot_id)
        if event_type:
            clauses.append("event_type=?")
            params.append(event_type)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        rows = self._db.fetchall(sql, tuple(params))
        return [
            AuditEventRecord(
                id=int(row["id"]),
                timestamp=from_iso(row["timestamp"]),
                level=row["level"],
                event_type=row["event_type"],
                bot_id=row["bot_id"],
                summary=row["summary"],
                payload=dict(from_json(row["payload_json"])),
            )
            for row in rows
        ]


class MemoryRepository:
    def __init__(self, db: SQLiteDB) -> None:
        self._db = db

    def add_episode(self, episode: MemoryEpisodeRecord) -> None:
        self._db.execute(
            """
            INSERT INTO memory_episodes(id, bot_id, event_type, content, metadata_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                episode.id,
                episode.bot_id,
                episode.event_type,
                episode.content,
                to_json(episode.metadata),
                to_iso(episode.created_at),
            ),
        )

    def count_episodes(self, *, bot_id: str) -> int:
        row = self._db.fetchone("SELECT COUNT(*) AS c FROM memory_episodes WHERE bot_id=?", (bot_id,))
        return int(row["c"]) if row else 0

    def recent_episodes(self, *, bot_id: str, limit: int) -> list[MemoryEpisodeRecord]:
        rows = self._db.fetchall(
            "SELECT * FROM memory_episodes WHERE bot_id=? ORDER BY created_at DESC LIMIT ?",
            (bot_id, limit),
        )
        return [
            MemoryEpisodeRecord(
                id=row["id"],
                bot_id=row["bot_id"],
                event_type=row["event_type"],
                content=row["content"],
                metadata=dict(from_json(row["metadata_json"])),
                created_at=from_iso(row["created_at"]),
            )
            for row in rows
        ]

    def add_semantic(self, record: MemorySemanticRecord) -> None:
        self._db.transaction(
            (
                (
                    """
                    INSERT INTO memory_semantic_records(
                        id, bot_id, source, content, lexical_signature, metadata_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        bot_id=excluded.bot_id,
                        source=excluded.source,
                        content=excluded.content,
                        lexical_signature=excluded.lexical_signature,
                        metadata_json=excluded.metadata_json,
                        created_at=excluded.created_at
                    """,
                    (
                        record.id,
                        record.bot_id,
                        record.source,
                        record.content,
                        record.lexical_signature,
                        to_json(record.metadata),
                        to_iso(record.created_at),
                    ),
                ),
                (
                    """
                    INSERT INTO memory_embeddings(memory_id, dimensions, vector_json, norm)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(memory_id) DO UPDATE SET
                        dimensions=excluded.dimensions,
                        vector_json=excluded.vector_json,
                        norm=excluded.norm
                    """,
                    (record.id, int(record.dimensions), to_json(record.vector), float(record.norm)),
                ),
            )
        )

    def semantic_candidates(
        self,
        *,
        bot_id: str,
        lexical_tokens: list[str],
        limit: int,
    ) -> list[MemorySemanticRecord]:
        sql = """
            SELECT r.*, e.dimensions, e.vector_json, e.norm
            FROM memory_semantic_records r
            JOIN memory_embeddings e ON e.memory_id = r.id
            WHERE r.bot_id = ?
        """
        params: list[object] = [bot_id]
        if lexical_tokens:
            token_clauses: list[str] = []
            for token in lexical_tokens[:8]:
                token_clauses.append("r.lexical_signature LIKE ?")
                params.append(f"% {token} %")
            sql += " AND (" + " OR ".join(token_clauses) + ")"
        sql += " ORDER BY r.created_at DESC LIMIT ?"
        params.append(limit)

        rows = self._db.fetchall(sql, tuple(params))
        if not rows and lexical_tokens:
            rows = self._db.fetchall(
                """
                SELECT r.*, e.dimensions, e.vector_json, e.norm
                FROM memory_semantic_records r
                JOIN memory_embeddings e ON e.memory_id = r.id
                WHERE r.bot_id = ?
                ORDER BY r.created_at DESC
                LIMIT ?
                """,
                (bot_id, limit),
            )
        return [
            MemorySemanticRecord(
                id=row["id"],
                bot_id=row["bot_id"],
                source=row["source"],
                content=row["content"],
                lexical_signature=row["lexical_signature"],
                metadata=dict(from_json(row["metadata_json"])),
                created_at=from_iso(row["created_at"]),
                dimensions=int(row["dimensions"]),
                vector=list(from_json(row["vector_json"])),
                norm=float(row["norm"]),
            )
            for row in rows
        ]

    def count_semantic(self, *, bot_id: str) -> int:
        row = self._db.fetchone("SELECT COUNT(*) AS c FROM memory_semantic_records WHERE bot_id=?", (bot_id,))
        return int(row["c"]) if row else 0


@dataclass(slots=True)
class SidecarRepositories:
    bots: BotRepository
    snapshots: SnapshotRepository
    actions: ActionRepository
    telemetry: TelemetryRepository
    macros: MacroRepository
    audit: AuditRepository
    memory: MemoryRepository


def create_repositories(
    *,
    db: SQLiteDB,
    snapshot_history_per_bot: int,
    telemetry_max_per_bot: int,
    telemetry_operational_window_minutes: int,
    audit_history: int,
) -> SidecarRepositories:
    return SidecarRepositories(
        bots=BotRepository(db),
        snapshots=SnapshotRepository(db, max_history_per_bot=snapshot_history_per_bot),
        actions=ActionRepository(db),
        telemetry=TelemetryRepository(
            db,
            max_per_bot=telemetry_max_per_bot,
            operational_window_minutes=telemetry_operational_window_minutes,
        ),
        macros=MacroRepository(db),
        audit=AuditRepository(db, max_history=audit_history),
        memory=MemoryRepository(db),
    )
