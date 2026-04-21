from __future__ import annotations

import logging
from dataclasses import dataclass, field
from threading import RLock

from ai_sidecar.contracts.ml_subconscious import DecisionSource, MLTrainingEpisode, ModelFamily
from ai_sidecar.ml_subconscious.model_registry import vectorize_state_features

logger = logging.getLogger(__name__)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


@dataclass(slots=True)
class LabelingPipeline:
    min_confidence: float = 0.55
    max_labels: int = 50000
    _lock: RLock = field(default_factory=RLock)
    _labels: list[dict[str, object]] = field(default_factory=list)
    _counters: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._counters = {
            "labels_total": 0,
            "labels_llm": 0,
            "labels_rule": 0,
            "labels_dropped_low_conf": 0,
            "labels_dropped_noise": 0,
        }

    def _confidence(self, episode: MLTrainingEpisode) -> float:
        base = 0.35
        if episode.decision_source == DecisionSource.llm:
            base = 0.72
        elif episode.decision_source == DecisionSource.rule:
            payload_conf = episode.decision_payload.get("rule_confidence")
            if isinstance(payload_conf, (int, float)):
                base = float(payload_conf)
            else:
                base = 0.82
        elif episode.decision_source == DecisionSource.ml:
            base = 0.6

        if episode.outcome.success:
            base += 0.12
        else:
            base -= 0.25

        base += float(episode.outcome.reward) * 0.2
        if episode.safety_flags:
            base -= min(0.3, len(episode.safety_flags) * 0.06)

        latency = float(episode.outcome.latency_ms or 0.0)
        if latency > 2000.0:
            base -= 0.08

        if "death" in {item.lower() for item in episode.outcome.side_effects}:
            base -= 0.25
        return _clamp01(base)

    def _family_label_rows(self, episode: MLTrainingEpisode, confidence: float) -> list[dict[str, object]]:
        payload = dict(episode.decision_payload or {})
        action = dict(episode.executed_action or {})
        state = dict(episode.state_features or {})
        vector, feature_contrib = vectorize_state_features(state)

        rows: list[dict[str, object]] = []
        encounter_profile = str(payload.get("combat_profile") or action.get("combat_profile") or payload.get("profile") or "balanced")
        rows.append(
            {
                "episode_id": episode.episode_id,
                "bot_id": episode.bot_id,
                "family": ModelFamily.encounter_classifier.value,
                "label": encounter_profile,
                "target": 1.0 if episode.outcome.success else 0.0,
                "confidence": confidence,
                "vector": vector,
                "features": feature_contrib,
                "context": {"decision_source": episode.decision_source.value},
            }
        )

        loot_priority = float(payload.get("loot_priority") or payload.get("loot_score") or (0.8 if episode.outcome.success else 0.2))
        loot_item = str(payload.get("loot_item") or action.get("loot_item") or "unknown")
        rows.append(
            {
                "episode_id": episode.episode_id,
                "bot_id": episode.bot_id,
                "family": ModelFamily.loot_ranker.value,
                "label": loot_item,
                "target": max(0.0, min(1.0, loot_priority)),
                "confidence": confidence,
                "vector": vector,
                "features": feature_contrib,
                "context": {"decision_source": episode.decision_source.value},
            }
        )

        route_strategy = str(payload.get("route_recovery") or payload.get("stuck_strategy") or "repath")
        rows.append(
            {
                "episode_id": episode.episode_id,
                "bot_id": episode.bot_id,
                "family": ModelFamily.route_recovery_classifier.value,
                "label": route_strategy,
                "target": 1.0 if episode.outcome.success else 0.0,
                "confidence": confidence,
                "vector": vector,
                "features": feature_contrib,
                "context": {"decision_source": episode.decision_source.value},
            }
        )

        npc_branch = str(payload.get("npc_branch") or payload.get("dialogue_branch") or "default")
        rows.append(
            {
                "episode_id": episode.episode_id,
                "bot_id": episode.bot_id,
                "family": ModelFamily.npc_dialogue_predictor.value,
                "label": npc_branch,
                "target": 1.0 if episode.outcome.success else 0.0,
                "confidence": confidence,
                "vector": vector,
                "features": feature_contrib,
                "context": {"decision_source": episode.decision_source.value},
            }
        )

        anomaly = 1.0 if (not episode.outcome.success or "death" in {item.lower() for item in episode.outcome.side_effects}) else 0.0
        rows.append(
            {
                "episode_id": episode.episode_id,
                "bot_id": episode.bot_id,
                "family": ModelFamily.risk_anomaly_detector.value,
                "label": "anomaly" if anomaly > 0.0 else "normal",
                "target": anomaly,
                "confidence": confidence,
                "vector": vector,
                "features": feature_contrib,
                "context": {"decision_source": episode.decision_source.value},
            }
        )

        memory_candidate = str(payload.get("memory_id") or payload.get("memory_hint") or "none")
        rows.append(
            {
                "episode_id": episode.episode_id,
                "bot_id": episode.bot_id,
                "family": ModelFamily.memory_retrieval_ranker.value,
                "label": memory_candidate,
                "target": max(0.0, min(1.0, 0.5 + 0.5 * float(episode.outcome.reward))),
                "confidence": confidence,
                "vector": vector,
                "features": feature_contrib,
                "context": {
                    "decision_source": episode.decision_source.value,
                    "query": str(payload.get("memory_query") or ""),
                },
            }
        )
        return rows

    def _dedupe_noise(self, labels: list[dict[str, object]]) -> list[dict[str, object]]:
        best: dict[tuple[str, str, str], dict[str, object]] = {}
        for item in labels:
            key = (str(item.get("episode_id")), str(item.get("family")), str(item.get("label")))
            prev = best.get(key)
            if prev is None:
                best[key] = item
                continue
            prev_conf = float(prev.get("confidence") or 0.0)
            cur_conf = float(item.get("confidence") or 0.0)
            if cur_conf > prev_conf:
                best[key] = item
        return list(best.values())

    def label_episode(self, episode: MLTrainingEpisode) -> list[dict[str, object]]:
        confidence = self._confidence(episode)
        if confidence < self.min_confidence:
            with self._lock:
                self._counters["labels_dropped_low_conf"] += 1
            return []

        if episode.decision_source == DecisionSource.rule and not episode.outcome.success:
            with self._lock:
                self._counters["labels_dropped_noise"] += 1
            return []

        rows = self._family_label_rows(episode, confidence)
        rows = self._dedupe_noise(rows)
        if not rows:
            return []

        with self._lock:
            self._labels.extend(rows)
            if len(self._labels) > self.max_labels:
                self._labels = self._labels[-self.max_labels :]
            self._counters["labels_total"] += len(rows)
            if episode.decision_source == DecisionSource.llm:
                self._counters["labels_llm"] += len(rows)
            elif episode.decision_source == DecisionSource.rule:
                self._counters["labels_rule"] += len(rows)

        logger.info(
            "ml_labels_generated",
            extra={
                "event": "ml_labels_generated",
                "episode_id": episode.episode_id,
                "bot_id": episode.bot_id,
                "rows": len(rows),
                "confidence": confidence,
            },
        )
        return rows

    def replay_buffer(self, *, family: ModelFamily, bot_id: str | None = None, limit: int = 5000) -> list[dict[str, object]]:
        with self._lock:
            rows = list(self._labels)
        rows = [item for item in rows if str(item.get("family")) == family.value]
        if bot_id:
            rows = [item for item in rows if str(item.get("bot_id")) == bot_id]
        return rows[-max(1, int(limit)) :]

    def counters(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counters)

