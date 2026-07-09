"""Extended ShadowModeEvaluator with cross-bot experience sharing."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import RLock
from typing import Any

from ai_sidecar.contracts.ml_subconscious import ModelFamily

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ShadowModeEvaluator:
    """Compares planner decisions (what the bot chose) with ML model recommendations.

    Extended with cross-bot experience sharing: when a bot's planner disagrees
    with its own model but another bot's experience supports a different choice,
    that cross-bot signal is recorded as an additional comparison dimension.
    """

    max_records: int = 50000
    _lock: RLock = field(default_factory=RLock)
    _records: list[dict[str, object]] = field(default_factory=list)
    _by_family: dict[str, dict[str, float]] = field(default_factory=dict)
    _cross_bot_insights: list[dict[str, object]] = field(default_factory=list)

    def _normalize_choice(
        self, family: ModelFamily, planner_choice: dict[str, object], recommendation: dict[str, object]
    ) -> tuple[str, str]:
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
        elif family == ModelFamily.heuristic_decision:
            planned = str(planner_choice.get("goal") or "")
            predicted = str(recommendation.get("action") or recommendation.get("top_domain") or "")
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
        cross_bot_recommendation: dict[str, object] | None = None,
    ) -> dict[str, object]:
        """Standard shadow-mode comparison, plus optional cross-bot insight."""
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

        # Cross-bot experience dimension
        cross_bot_match = False
        cross_bot_action = ""
        if cross_bot_recommendation and cross_bot_recommendation.get("has_cross_bot_data"):
            cross_bot_action = str(cross_bot_recommendation.get("best_action") or "")
            if cross_bot_action and planned == cross_bot_action:
                cross_bot_match = True
            record["cross_bot_action"] = cross_bot_action
            record["cross_bot_match"] = cross_bot_match
            record["cross_bot_peers"] = int(cross_bot_recommendation.get("peer_count", 0))
            record["cross_bot_rate"] = float(cross_bot_recommendation.get("success_rate", 0.0))

        with self._lock:
            self._records.append(record)
            if len(self._records) > self.max_records:
                self._records = self._records[-self.max_records:]

            stats = self._by_family.setdefault(
                family.value,
                {
                    "total": 0.0,
                    "matched": 0.0,
                    "confidence_sum": 0.0,
                    "high_conf_disagreements": 0.0,
                    "cross_bot_total": 0.0,
                    "cross_bot_matched": 0.0,
                },
            )
            stats["total"] += 1.0
            stats["matched"] += 1.0 if matched else 0.0
            stats["confidence_sum"] += float(record["confidence"])
            if not matched and float(record["confidence"]) >= 0.75:
                stats["high_conf_disagreements"] += 1.0

            if cross_bot_action:
                stats["cross_bot_total"] += 1.0
                stats["cross_bot_matched"] += 1.0 if cross_bot_match else 0.0

            # Store cross-bot insight separately
            if cross_bot_action and cross_bot_action != planned:
                insight = {
                    "observed_at": datetime.now(UTC),
                    "bot_id": bot_id,
                    "family": family.value,
                    "planner_choice": planned,
                    "cross_bot_recommendation": cross_bot_action,
                    "cross_bot_success_rate": float(cross_bot_recommendation.get("success_rate", 0.0)),
                    "peer_count": int(cross_bot_recommendation.get("peer_count", 0)),
                }
                self._cross_bot_insights.append(insight)
                if len(self._cross_bot_insights) > 1000:
                    self._cross_bot_insights = self._cross_bot_insights[-1000:]

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
                "cross_bot_action": cross_bot_action,
                "cross_bot_match": cross_bot_match,
            },
        )

        return {
            "matched": matched,
            "planned": planned,
            "predicted": predicted,
            "confidence": float(record["confidence"]),
            "mode": "shadow_only",
            "cross_bot_action": cross_bot_action or "",
            "cross_bot_match": cross_bot_match,
        }

    def metrics(self) -> dict[str, object]:
        with self._lock:
            by_family = {key: dict(value) for key, value in self._by_family.items()}
            total_records = len(self._records)
            total_cross_bot = len(self._cross_bot_insights)

        rows: dict[str, object] = {}
        for family, stats in by_family.items():
            total = float(stats.get("total") or 0.0)
            matched = float(stats.get("matched") or 0.0)
            confidence_sum = float(stats.get("confidence_sum") or 0.0)
            cb_total = float(stats.get("cross_bot_total") or 0.0)
            cb_matched = float(stats.get("cross_bot_matched") or 0.0)
            rows[family] = {
                "total": int(total),
                "matched": int(matched),
                "match_rate": (matched / total) if total > 0 else 0.0,
                "confidence_mean": (confidence_sum / total) if total > 0 else 0.0,
                "high_conf_disagreements": int(stats.get("high_conf_disagreements") or 0.0),
                "cross_bot_comparisons": int(cb_total),
                "cross_bot_agreement_rate": (cb_matched / cb_total) if cb_total > 0 else 0.0,
                "cross_bot_insights": total_cross_bot,
            }
        return {
            "total_records": total_records,
            "cross_bot_insights": total_cross_bot,
            "by_family": rows,
        }

    def recent(
        self,
        *,
        family: ModelFamily | None = None,
        limit: int = 100,
        include_cross_bot: bool = False,
    ) -> list[dict[str, object]]:
        with self._lock:
            rows = list(self._records)
        if family is not None:
            rows = [item for item in rows if str(item.get("family")) == family.value]
        result = rows[-max(1, int(limit)) :]
        if include_cross_bot:
            with self._lock:
                cb = list(self._cross_bot_insights)
            result.extend(cb[-max(1, int(limit // 2)):])
        return result
