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
    QuestState,
    RiskState,
    SocialState,
)


@dataclass(slots=True)
class _BotProjection:
    operational: BotOperationalState
    encounter: EncounterState = field(default_factory=EncounterState)
    navigation: NavigationState = field(default_factory=NavigationState)
    quest: QuestState = field(default_factory=QuestState)
    inventory: InventoryState = field(default_factory=InventoryState)
    economy: EconomyState = field(default_factory=EconomyState)
    social: SocialState = field(default_factory=SocialState)
    risk: RiskState = field(default_factory=RiskState)
    macro_execution: MacroExecutionState = field(default_factory=MacroExecutionState)
    fleet_intent: FleetIntentState = field(default_factory=FleetIntentState)
    recent_event_ids: list[str] = field(default_factory=list)


class WorldStateProjector:
    def __init__(self) -> None:
        self._lock = RLock()
        self._by_bot: dict[str, _BotProjection] = {}

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
            elif event.event_family == EventFamily.telemetry:
                self._apply_telemetry(projection, event, now)

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
        projection.operational.raw = payload.get("raw") if isinstance(payload.get("raw"), dict) else {}

        projection.navigation.map = position.get("map")
        projection.navigation.x = position.get("x")
        projection.navigation.y = position.get("y")
        projection.navigation.route_status = "moving" if projection.operational.ai_sequence == "route" else "idle"
        projection.navigation.updated_at = now

        projection.inventory.zeny = inventory.get("zeny")
        projection.inventory.item_count = int(inventory.get("item_count") or 0)
        projection.inventory.weight = vitals.get("weight")
        projection.inventory.weight_max = vitals.get("weight_max")
        if projection.inventory.weight is not None and projection.inventory.weight_max:
            projection.inventory.overweight_ratio = float(projection.inventory.weight) / float(projection.inventory.weight_max)
        projection.inventory.updated_at = now

        projection.economy.zeny = inventory.get("zeny")
        projection.economy.updated_at = now

        projection.encounter.in_encounter = bool(combat.get("is_in_combat"))
        projection.encounter.target_id = combat.get("target_id")
        projection.encounter.updated_at = now

        hp = float(vitals.get("hp") or 0.0)
        hp_max = float(vitals.get("hp_max") or 0.0)
        hp_ratio = (hp / hp_max) if hp_max > 0 else 1.0
        projection.risk.death_risk_score = max(0.0, 1.0 - hp_ratio)
        projection.risk.danger_score = max(projection.risk.danger_score, projection.risk.death_risk_score)
        projection.risk.updated_at = now

    def _apply_actor(self, projection: _BotProjection, event: NormalizedEvent, now: datetime) -> None:
        payload = event.payload
        relation = str(payload.get("relation") or "")
        actor_type = str(payload.get("actor_type") or "")
        if relation in {"hostile", "monster", "enemy"} or actor_type == "monster":
            projection.encounter.nearby_hostiles = max(0, projection.encounter.nearby_hostiles + 1)
        elif relation in {"party", "ally", "friend"}:
            projection.encounter.nearby_allies = max(0, projection.encounter.nearby_allies + 1)
        projection.encounter.risk_score = float(projection.encounter.nearby_hostiles) / float(max(1, projection.encounter.nearby_allies + 1))
        projection.encounter.updated_at = now

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

    def _apply_config(self, projection: _BotProjection, event: NormalizedEvent, now: datetime) -> None:
        key = event.tags.get("key")
        if key:
            projection.fleet_intent.constraints[f"config.{key}"] = event.payload.get("value")
        projection.fleet_intent.updated_at = now

    def _apply_quest(self, projection: _BotProjection, event: NormalizedEvent, now: datetime) -> None:
        payload = event.payload
        quest_id = str(payload.get("quest_id") or "")
        state_to = str(payload.get("state_to") or "")
        if quest_id:
            if quest_id not in projection.quest.active_quests and state_to not in {"completed", "abandoned"}:
                projection.quest.active_quests.append(quest_id)
            projection.quest.quest_status[quest_id] = state_to or "unknown"
        npc = payload.get("npc")
        if isinstance(npc, str) and npc:
            projection.quest.last_npc = npc
        projection.quest.updated_at = now

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

    def _apply_telemetry(self, projection: _BotProjection, event: NormalizedEvent, now: datetime) -> None:
        if event.severity.value in {"warning", "error", "critical"}:
            projection.risk.anomaly_flags.append(event.event_type)
            if len(projection.risk.anomaly_flags) > 64:
                del projection.risk.anomaly_flags[: len(projection.risk.anomaly_flags) - 64]
            base = 0.2 if event.severity.value == "warning" else 0.4
            projection.risk.danger_score = min(1.0, projection.risk.danger_score + base)
            projection.risk.updated_at = now

    def _normalize_dt(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

