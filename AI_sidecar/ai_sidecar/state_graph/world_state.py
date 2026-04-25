from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import RLock

from ai_sidecar.contracts.events import EventFamily, NormalizedEvent
from ai_sidecar.contracts.state_graph import (
    BotOperationalState,
    EconomyState,
    EncounterState,
    FleetIntentState,
    InventoryState,
    LearningFeatureVector,
    MacroExecutionState,
    NavigationState,
    NpcState,
    QuestState,
    RiskState,
    SocialState,
)
from ai_sidecar.state_graph.economy_tracker import EconomyTracker
from ai_sidecar.state_graph.npc_tracker import NpcInteractionTracker
from ai_sidecar.state_graph.quest_tracker import QuestProgressTracker


@dataclass(slots=True)
class _BotProjection:
    operational: BotOperationalState
    encounter: EncounterState = field(default_factory=EncounterState)
    navigation: NavigationState = field(default_factory=NavigationState)
    quest: QuestState = field(default_factory=QuestState)
    inventory: InventoryState = field(default_factory=InventoryState)
    economy: EconomyState = field(default_factory=EconomyState)
    social: SocialState = field(default_factory=SocialState)
    npc: NpcState = field(default_factory=NpcState)
    risk: RiskState = field(default_factory=RiskState)
    macro_execution: MacroExecutionState = field(default_factory=MacroExecutionState)
    fleet_intent: FleetIntentState = field(default_factory=FleetIntentState)
    recent_event_ids: list[str] = field(default_factory=list)
    actor_relations: dict[str, tuple[str, str]] = field(default_factory=dict)
    last_in_game: bool | None = None
    last_disconnect_at: datetime | None = None
    last_map_name: str | None = None
    last_route_signature: tuple[str | None, int | None, int | None] | None = None


def _int_or_none(value: object) -> int | None:
    try:
        return int(value) if value is not None else None  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _str_or_none(value: object) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    return s if s else None


class WorldStateProjector:
    def __init__(self) -> None:
        self._lock = RLock()
        self._by_bot: dict[str, _BotProjection] = {}
        self._npc_tracker = NpcInteractionTracker()
        self._economy_tracker = EconomyTracker()
        self._quest_tracker = QuestProgressTracker()

    def observe_event(self, event: NormalizedEvent) -> None:
        now = self._normalize_dt(event.observed_at)
        bot_id = event.meta.bot_id

        with self._lock:
            projection = self._by_bot.setdefault(
                bot_id,
                _BotProjection(operational=BotOperationalState(bot_id=bot_id, updated_at=now)),
            )
            projection.recent_event_ids.append(event.event_id)
            if len(projection.recent_event_ids) > 128:
                del projection.recent_event_ids[: len(projection.recent_event_ids) - 128]

            if event.event_family == EventFamily.snapshot:
                self._apply_snapshot(projection, event, now)
            elif event.event_family == EventFamily.actor_state:
                self._apply_actor(projection, event, now)
            elif event.event_family == EventFamily.chat:
                self._apply_chat(projection, event, now)
            elif event.event_family == EventFamily.config:
                self._apply_config(projection, event, now)
            elif event.event_family == EventFamily.quest:
                self._apply_quest(projection, event, now)
            elif event.event_family == EventFamily.macro:
                self._apply_macro(projection, event, now)
            elif event.event_family == EventFamily.action:
                self._apply_action(projection, event, now)
            elif event.event_family == EventFamily.telemetry:
                self._apply_telemetry(projection, event, now)
            elif event.event_family == EventFamily.lifecycle:
                self._apply_lifecycle(projection, event, now)

    def export(self, *, bot_id: str, features: LearningFeatureVector) -> dict[str, object]:
        with self._lock:
            projection = self._by_bot.get(bot_id)
            if projection is None:
                projection = _BotProjection(operational=BotOperationalState(bot_id=bot_id, updated_at=datetime.now(UTC)))
            return {
                "operational": projection.operational,
                "encounter": projection.encounter,
                "navigation": projection.navigation,
                "quest": projection.quest,
                "inventory": projection.inventory,
                "economy": projection.economy,
                "social": projection.social,
                "npc": projection.npc,
                "risk": projection.risk,
                "macro_execution": projection.macro_execution,
                "fleet_intent": projection.fleet_intent,
                "recent_event_ids": list(projection.recent_event_ids),
                "features": features,
            }

    def _apply_snapshot(self, projection: _BotProjection, event: NormalizedEvent, now: datetime) -> None:
        payload = event.payload
        position = payload.get("position") if isinstance(payload.get("position"), dict) else {}
        vitals = payload.get("vitals") if isinstance(payload.get("vitals"), dict) else {}
        combat = payload.get("combat") if isinstance(payload.get("combat"), dict) else {}
        inventory = payload.get("inventory") if isinstance(payload.get("inventory"), dict) else {}
        raw_payload = payload.get("raw") if isinstance(payload.get("raw"), dict) else {}

        projection.operational.map = position.get("map")
        projection.operational.x = position.get("x")
        projection.operational.y = position.get("y")
        projection.operational.hp = vitals.get("hp")
        projection.operational.hp_max = vitals.get("hp_max")
        projection.operational.sp = vitals.get("sp")
        projection.operational.sp_max = vitals.get("sp_max")
        projection.operational.ai_sequence = combat.get("ai_sequence")
        projection.operational.in_combat = bool(combat.get("is_in_combat"))
        projection.operational.target_id = combat.get("target_id")
        projection.operational.updated_at = now
        projection.operational.raw = raw_payload

        in_game_value = raw_payload.get("in_game")
        in_game: bool | None = None
        if isinstance(in_game_value, bool):
            in_game = in_game_value
        elif isinstance(in_game_value, (int, float)):
            in_game = bool(in_game_value)
        elif isinstance(in_game_value, str):
            in_game = in_game_value.strip().lower() in {"1", "true", "yes", "on"}

        if in_game is not None:
            projection.last_in_game = in_game
            projection.operational.liveness_state = "online" if in_game else "disconnected"

        reconnect_age = raw_payload.get("reconnect_age_s")
        if isinstance(reconnect_age, (int, float)):
            projection.operational.reconnect_age_s = max(0.0, float(reconnect_age))
        elif isinstance(reconnect_age, str):
            try:
                projection.operational.reconnect_age_s = max(0.0, float(reconnect_age))
            except ValueError:
                pass

        death_count = raw_payload.get("death_count")
        if isinstance(death_count, int):
            projection.operational.death_count = max(0, death_count)

        respawn_state = _str_or_none(raw_payload.get("respawn_state"))
        if respawn_state:
            projection.operational.respawn_state = respawn_state

        # Populate progression fields from enriched snapshot (bridge v2+).
        progression = payload.get("progression") if isinstance(payload.get("progression"), dict) else {}
        if progression:
            projection.operational.job_id = _int_or_none(progression.get("job_id"))
            projection.operational.job_name = _str_or_none(progression.get("job_name"))
            projection.operational.base_level = _int_or_none(progression.get("base_level"))
            projection.operational.job_level = _int_or_none(progression.get("job_level"))
            projection.operational.base_exp = _int_or_none(progression.get("base_exp"))
            projection.operational.base_exp_max = _int_or_none(progression.get("base_exp_max"))
            projection.operational.job_exp = _int_or_none(progression.get("job_exp"))
            projection.operational.job_exp_max = _int_or_none(progression.get("job_exp_max"))
            projection.operational.skill_points = _int_or_none(progression.get("skill_points"))
            projection.operational.stat_points = _int_or_none(progression.get("stat_points"))

        skills_payload = payload.get("skills")
        if isinstance(skills_payload, list):
            skill_map: dict[str, int] = {}
            for item in skills_payload:
                if not isinstance(item, dict):
                    continue
                skill_name = _str_or_none(item.get("skill_name") or item.get("skill_id"))
                if not skill_name:
                    continue
                skill_map[skill_name] = int(_int_or_none(item.get("level")) or 0)
            projection.operational.skills = skill_map
            projection.operational.skill_count = len(skill_map)

        projection.navigation.map = position.get("map")
        projection.navigation.x = position.get("x")
        projection.navigation.y = position.get("y")
        ai_seq = (projection.operational.ai_sequence or "").strip().lower()
        if ai_seq.startswith("route") or ai_seq.startswith("move"):
            projection.navigation.route_status = "moving"
        elif ai_seq:
            projection.navigation.route_status = ai_seq
        else:
            projection.navigation.route_status = "idle"

        route_signature = (
            _str_or_none(position.get("map")),
            _int_or_none(position.get("x")),
            _int_or_none(position.get("y")),
        )
        moving_now = projection.navigation.route_status == "moving"
        if moving_now and projection.last_route_signature == route_signature:
            projection.navigation.route_churn_count += 1
        projection.last_route_signature = route_signature

        route_failure_count = raw_payload.get("route_failure_count")
        if isinstance(route_failure_count, int):
            projection.navigation.route_failure_count = max(projection.navigation.route_failure_count, route_failure_count)

        projection.navigation.updated_at = now

        projection.inventory.zeny = inventory.get("zeny")
        projection.inventory.item_count = int(inventory.get("item_count") or 0)
        projection.inventory.weight = vitals.get("weight")
        projection.inventory.weight_max = vitals.get("weight_max")
        if projection.inventory.weight is not None and projection.inventory.weight_max:
            projection.inventory.overweight_ratio = float(projection.inventory.weight) / float(projection.inventory.weight_max)
            projection.inventory.weight_pressure = max(
                0.0,
                min(1.0, (projection.inventory.overweight_ratio - 0.5) / 0.5),
            )
        projection.inventory.updated_at = now

        inventory_items_payload = payload.get("inventory_items")
        if isinstance(inventory_items_payload, list):
            consumables: dict[str, int] = {}
            for item in inventory_items_payload[:128]:
                if not isinstance(item, dict):
                    continue
                category = str(item.get("category") or "").strip().lower()
                if category not in {"consumable", "potion", "ammo", "healing"}:
                    continue
                name = str(item.get("name") or item.get("item_id") or "").strip()
                if not name:
                    continue
                qty = int(_int_or_none(item.get("quantity")) or 0)
                if qty > 0:
                    consumables[name] = qty
            projection.inventory.consumables = consumables

        total_consumables = sum(int(max(0, qty)) for qty in projection.inventory.consumables.values())
        if total_consumables <= 0:
            projection.inventory.consumable_depletion_score = 1.0
        elif total_consumables <= 3:
            projection.inventory.consumable_depletion_score = 0.9
        elif total_consumables <= 10:
            projection.inventory.consumable_depletion_score = 0.6
        elif total_consumables <= 25:
            projection.inventory.consumable_depletion_score = 0.3
        else:
            projection.inventory.consumable_depletion_score = 0.0

        self._economy_tracker.observe_snapshot(bot_id=projection.operational.bot_id, payload=payload, observed_at=now)
        projection.economy = self._economy_tracker.export(
            bot_id=projection.operational.bot_id,
            zeny=_int_or_none(inventory.get("zeny")),
            observed_at=now,
        )

        projection.encounter.in_encounter = bool(combat.get("is_in_combat"))
        projection.encounter.target_id = combat.get("target_id")
        projection.encounter.updated_at = now

        self._npc_tracker.observe_snapshot(bot_id=projection.operational.bot_id, payload=payload, observed_at=now)
        projection.npc = self._npc_tracker.export(bot_id=projection.operational.bot_id, observed_at=now)

        self._quest_tracker.observe_snapshot(bot_id=projection.operational.bot_id, payload=payload, observed_at=now)
        projection.quest = self._quest_tracker.export(bot_id=projection.operational.bot_id, observed_at=now)

        # Rebuild canonical actor relation cache from inlined actors array (if bridge provided them).
        actors_list = payload.get("actors")
        if isinstance(actors_list, list):
            relation_map: dict[str, tuple[str, str]] = {}
            for actor in actors_list:
                if not isinstance(actor, dict):
                    continue
                actor_id = str(actor.get("actor_id") or "").strip()
                if not actor_id:
                    continue
                relation_map[actor_id] = (
                    str(actor.get("relation") or "").strip().lower(),
                    str(actor.get("actor_type") or "").strip().lower(),
                )
            projection.actor_relations = relation_map
            self._recompute_actor_encounter(projection, now)

        hp = float(vitals.get("hp") or 0.0)
        hp_max = float(vitals.get("hp_max") or 0.0)
        hp_ratio = (hp / hp_max) if hp_max > 0 else 1.0
        if hp_max > 0 and hp <= 0:
            projection.operational.respawn_state = "dead"
            projection.operational.death_count = max(1, projection.operational.death_count)
        elif hp_max > 0 and hp > 0 and projection.operational.respawn_state in {"dead", "respawning"}:
            projection.operational.respawn_state = "alive"
        projection.risk.death_risk_score = max(0.0, 1.0 - hp_ratio)
        projection.risk.danger_score = max(projection.risk.danger_score, projection.risk.death_risk_score)
        projection.risk.updated_at = now

    def _apply_lifecycle(self, projection: _BotProjection, event: NormalizedEvent, now: datetime) -> None:
        event_type = str(event.event_type or "").strip().lower()
        payload = event.payload if isinstance(event.payload, dict) else {}

        if event_type in {"lifecycle.disconnected", "lifecycle.disconnect"}:
            projection.operational.liveness_state = "disconnected"
            projection.last_in_game = False
            projection.last_disconnect_at = now
            projection.operational.reconnect_age_s = 0.0
        elif event_type in {"lifecycle.reconnected", "lifecycle.connected"}:
            projection.operational.liveness_state = "online"
            projection.last_in_game = True
            if projection.last_disconnect_at is not None:
                projection.operational.reconnect_age_s = max(0.0, (now - projection.last_disconnect_at).total_seconds())
        elif event_type == "lifecycle.death":
            projection.operational.death_count += 1
            projection.operational.respawn_state = "dead"
            projection.risk.death_risk_score = 1.0
            projection.risk.danger_score = max(projection.risk.danger_score, 0.95)
        elif event_type == "lifecycle.respawn":
            projection.operational.respawn_state = "respawned"
        elif event_type == "lifecycle.map_transfer":
            from_map = _str_or_none(payload.get("from_map"))
            to_map = _str_or_none(payload.get("to_map"))
            projection.navigation.destination_map = to_map
            projection.navigation.map = to_map or projection.navigation.map
            projection.last_map_name = to_map or projection.last_map_name
            projection.navigation.route_churn_count += 1
            if from_map and to_map and from_map != to_map:
                projection.navigation.route_status = "transferred"
        elif event_type in {"lifecycle.route_failure", "navigation.route_failure", "navigation.stuck"}:
            projection.navigation.route_failure_count += 1
            projection.navigation.stuck_score = min(1.0, projection.navigation.stuck_score + 0.2)
            projection.navigation.route_status = "failed"

        projection.operational.updated_at = now
        projection.navigation.updated_at = now
        projection.risk.updated_at = now

    def _apply_actor(self, projection: _BotProjection, event: NormalizedEvent, now: datetime) -> None:
        payload = event.payload
        actor_id = str(payload.get("actor_id") or "").strip()
        if not actor_id:
            return

        if event.event_type in {"actor.removed", "actor.disappeared"}:
            projection.actor_relations.pop(actor_id, None)
            self._npc_tracker.observe_event(event)
            projection.npc = self._npc_tracker.export(bot_id=projection.operational.bot_id, observed_at=now)
            self._recompute_actor_encounter(projection, now)
            return

        relation = str(payload.get("relation") or "").strip().lower()
        actor_type = str(payload.get("actor_type") or "").strip().lower()
        projection.actor_relations[actor_id] = (relation, actor_type)

        if actor_type == "npc":
            self._npc_tracker.observe_event(event)
            projection.npc = self._npc_tracker.export(bot_id=projection.operational.bot_id, observed_at=now)

        self._recompute_actor_encounter(projection, now)

    def _apply_chat(self, projection: _BotProjection, event: NormalizedEvent, now: datetime) -> None:
        projection.social.recent_chat_count += 1
        channel = str(event.tags.get("channel") or "")
        if channel == "private":
            projection.social.private_messages_5m += 1
        elif channel == "party":
            projection.social.party_messages_5m += 1
        elif channel == "guild":
            projection.social.guild_messages_5m += 1
        projection.social.last_interaction_at = now
        projection.social.updated_at = now

        self._npc_tracker.observe_event(event)
        projection.npc = self._npc_tracker.export(bot_id=projection.operational.bot_id, observed_at=now)

    def _apply_config(self, projection: _BotProjection, event: NormalizedEvent, now: datetime) -> None:
        key = event.tags.get("key")
        if key:
            projection.fleet_intent.constraints[f"config.{key}"] = event.payload.get("value")
        projection.fleet_intent.updated_at = now

    def _apply_quest(self, projection: _BotProjection, event: NormalizedEvent, now: datetime) -> None:
        self._quest_tracker.observe_event(event)
        projection.quest = self._quest_tracker.export(bot_id=projection.operational.bot_id, observed_at=now)

        self._npc_tracker.observe_event(event)
        projection.npc = self._npc_tracker.export(bot_id=projection.operational.bot_id, observed_at=now)

    def _apply_macro(self, projection: _BotProjection, event: NormalizedEvent, now: datetime) -> None:
        payload = event.payload
        action = str(payload.get("action") or "")
        macro_name = payload.get("macro_name")
        if action in {"start", "running"}:
            projection.macro_execution.macro_running = True
        elif action in {"stop", "finish", "failed"}:
            projection.macro_execution.macro_running = False
        if isinstance(macro_name, str):
            projection.macro_execution.macro_name = macro_name
        projection.macro_execution.last_result = action or event.event_type
        projection.macro_execution.updated_at = now

        self._economy_tracker.observe_event(event)
        projection.economy = self._economy_tracker.export(
            bot_id=projection.operational.bot_id,
            zeny=projection.economy.zeny,
            observed_at=now,
        )

    def _apply_telemetry(self, projection: _BotProjection, event: NormalizedEvent, now: datetime) -> None:
        self._economy_tracker.observe_event(event)
        projection.economy = self._economy_tracker.export(
            bot_id=projection.operational.bot_id,
            zeny=projection.economy.zeny,
            observed_at=now,
        )

        if event.severity.value in {"warning", "error", "critical"}:
            projection.risk.anomaly_flags.append(event.event_type)
            if len(projection.risk.anomaly_flags) > 64:
                del projection.risk.anomaly_flags[: len(projection.risk.anomaly_flags) - 64]
            base = 0.2 if event.severity.value == "warning" else 0.4
            projection.risk.danger_score = min(1.0, projection.risk.danger_score + base)
            projection.risk.updated_at = now

    def _apply_action(self, projection: _BotProjection, event: NormalizedEvent, now: datetime) -> None:
        self._economy_tracker.observe_event(event)
        projection.economy = self._economy_tracker.export(
            bot_id=projection.operational.bot_id,
            zeny=projection.economy.zeny,
            observed_at=now,
        )

    def _normalize_dt(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    def _recompute_actor_encounter(self, projection: _BotProjection, now: datetime) -> None:
        hostiles = 0
        allies = 0
        for relation, actor_type in projection.actor_relations.values():
            if relation in {"hostile", "monster", "enemy"} or actor_type == "monster":
                hostiles += 1
            elif relation in {"party", "ally", "friend"}:
                allies += 1

        projection.encounter.nearby_hostiles = hostiles
        projection.encounter.nearby_allies = allies
        projection.encounter.risk_score = float(hostiles) / float(max(1, allies + 1))
        projection.encounter.in_encounter = bool(projection.operational.in_combat or hostiles > 0)
        projection.encounter.updated_at = now
