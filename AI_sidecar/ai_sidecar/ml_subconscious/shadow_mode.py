from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import RLock

from ai_sidecar.contracts.ml_subconscious import ModelFamily

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ShadowModeEvaluator:
    max_records: int = 50000
    _lock: RLock = field(default_factory=RLock)
    _records: list[dict[str, object]] = field(default_factory=list)
    _by_family: dict[str, dict[str, float]] = field(default_factory=dict)

    def _normalize_choice(self, family: ModelFamily, planner_choice: dict[str, object], recommendation: dict[str, object]) -> tuple[str, str]:
        if family == ModelFamily.encounter_classifier:
            planned = str(planner_choice.get("combat_profile") or planner_choice.get("profile") or "")
            predicted = str(recommendation.get("combat_profile") or "")
        elif family == ModelFamily.loot_ranker:
            planned = str(planner_choice.get("loot_item") or planner_choice.get("top_loot") or "")
            predicted = str(recommendation.get("top_loot") or "")
        elif family == ModelFamily.route_recovery_classifier:
            planned = str(planner_choice.get("stuck_strategy") or planner_choice.get("route_recovery") or "")
            predicted = str(recommendation.get("stuck_strategy") or "")
        elif family == ModelFamily.npc_dialogue_predictor:
            planned = str(planner_choice.get("npc_branch") or planner_choice.get("next_branch") or "")
            predicted = str(recommendation.get("next_branch") or "")
        elif family == ModelFamily.risk_anomaly_detector:
            planned = str(planner_choice.get("risk_label") or planner_choice.get("anomaly") or "normal")
            predicted = str(recommendation.get("risk_label") or "normal")
        else:
            planned = str(planner_choice.get("memory_id") or planner_choice.get("top_memory") or "")
            predicted = str(recommendation.get("top_memory") or "")
        return planned, predicted

    def compare(
        self,
        *,
        bot_id: str,
        trace_id: str,
        family: ModelFamily,
        model_version: str,
        planner_choice: dict[str, object],
        recommendation: dict[str, object],
        confidence: float,
    ) -> dict[str, object]:
        planned, predicted = self._normalize_choice(family, planner_choice, recommendation)
        matched = bool(planned and predicted and planned == predicted)
        record = {
            "observed_at": datetime.now(UTC),
            "bot_id": bot_id,
            "trace_id": trace_id,
            "family": family.value,
            "model_version": model_version,
            "planner_choice": planner_choice,
            "recommendation": recommendation,
            "planned": planned,
            "predicted": predicted,
            "matched": matched,
            "confidence": float(max(0.0, min(1.0, confidence))),
        }

        with self._lock:
            self._records.append(record)
            if len(self._records) > self.max_records:
                self._records = self._records[-self.max_records :]

            stats = self._by_family.setdefault(
                family.value,
                {
                    "total": 0.0,
                    "matched": 0.0,
                    "confidence_sum": 0.0,
                    "high_conf_disagreements": 0.0,
                },
            )
            stats["total"] += 1.0
            stats["matched"] += 1.0 if matched else 0.0
            stats["confidence_sum"] += float(record["confidence"])
            if not matched and float(record["confidence"]) >= 0.75:
                stats["high_conf_disagreements"] += 1.0

        logger.info(
            "ml_shadow_comparison",
            extra={
                "event": "ml_shadow_comparison",
                "bot_id": bot_id,
                "trace_id": trace_id,
                "family": family.value,
                "model_version": model_version,
                "matched": matched,
                "confidence": confidence,
            },
        )

        return {
            "matched": matched,
            "planned": planned,
            "predicted": predicted,
            "confidence": float(record["confidence"]),
            "mode": "shadow_only",
        }

    def metrics(self) -> dict[str, object]:
        with self._lock:
            by_family = {key: dict(value) for key, value in self._by_family.items()}
            total_records = len(self._records)

        rows: dict[str, object] = {}
        for family, stats in by_family.items():
            total = float(stats.get("total") or 0.0)
            matched = float(stats.get("matched") or 0.0)
            confidence_sum = float(stats.get("confidence_sum") or 0.0)
            rows[family] = {
                "total": int(total),
                "matched": int(matched),
                "match_rate": (matched / total) if total > 0 else 0.0,
                "confidence_mean": (confidence_sum / total) if total > 0 else 0.0,
                "high_conf_disagreements": int(stats.get("high_conf_disagreements") or 0.0),
            }
        return {
            "total_records": total_records,
            "by_family": rows,
        }

    def recent(self, *, family: ModelFamily | None = None, limit: int = 100) -> list[dict[str, object]]:
        with self._lock:
            rows = list(self._records)
        if family is not None:
            rows = [item for item in rows if str(item.get("family")) == family.value]
        return rows[-max(1, int(limit)) :]

