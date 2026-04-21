from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock

from ai_sidecar.contracts.ml_subconscious import DecisionSource, MLTrainingEpisode

logger = logging.getLogger(__name__)


def _clamp(value: float, *, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _json_default(value: object) -> object:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


@dataclass(slots=True)
class ObservationCapture:
    workspace_root: Path
    max_in_memory: int = 20000
    _lock: RLock = field(default_factory=RLock)
    _episodes: list[MLTrainingEpisode] = field(default_factory=list)
    _counters: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._counters = {
            "observed_total": 0,
            "observed_llm": 0,
            "observed_rule": 0,
            "observed_ml": 0,
        }
        self._data_dir.mkdir(parents=True, exist_ok=True)

    @property
    def _data_dir(self) -> Path:
        return self.workspace_root / "AI_sidecar" / "data" / "ml_subconscious"

    @property
    def _episode_log_path(self) -> Path:
        stamp = datetime.now(UTC).strftime("%Y%m%d")
        return self._data_dir / f"episodes-{stamp}.jsonl"

    def _append_jsonl(self, payload: dict[str, object]) -> None:
        row = json.dumps(payload, ensure_ascii=False, default=_json_default)
        with self._episode_log_path.open("a", encoding="utf-8") as handle:
            handle.write(row + "\n")

    def _reward_breakdown(self, episode: MLTrainingEpisode) -> dict[str, float]:
        outcome = episode.outcome
        side_effects = {str(item).lower() for item in outcome.side_effects}
        safety_flags = {str(item).lower() for item in episode.safety_flags}
        payload = dict(episode.decision_payload or {})

        survival = 1.0 if outcome.success else -1.0
        if "death" in side_effects:
            survival = -1.0

        resource_eff = 0.5
        if "resource_waste" in side_effects or "overweight" in side_effects:
            resource_eff = -0.5
        resource_eff = float(payload.get("resource_efficiency", resource_eff)) if isinstance(payload.get("resource_efficiency"), (int, float)) else resource_eff

        objective_progress = 0.0
        if isinstance(payload.get("objective_progress"), (int, float)):
            objective_progress = _clamp(float(payload.get("objective_progress")), lo=-1.0, hi=1.0)
        elif outcome.success:
            objective_progress = 0.6

        social_safety = 1.0
        if "social_risk" in side_effects or "gm_attention" in side_effects:
            social_safety = -1.0

        rule_compliance = 1.0 if not safety_flags else -min(1.0, float(len(safety_flags)) / 4.0)

        latency_ms = float(outcome.latency_ms or 0.0)
        latency_efficiency = _clamp(1.0 - (latency_ms / 2000.0), lo=-1.0, hi=1.0)

        reduced_planner_cost = 1.0 if episode.decision_source in {DecisionSource.rule, DecisionSource.ml} else -0.2

        return {
            "survival": survival,
            "resource_efficiency": _clamp(resource_eff),
            "objective_progress": objective_progress,
            "social_safety": social_safety,
            "rule_compliance": rule_compliance,
            "latency_efficiency": latency_efficiency,
            "reduced_planner_cost": reduced_planner_cost,
        }

    def compute_reward(self, episode: MLTrainingEpisode) -> tuple[float, dict[str, float]]:
        breakdown = self._reward_breakdown(episode)
        weights = {
            "survival": 0.26,
            "resource_efficiency": 0.12,
            "objective_progress": 0.2,
            "social_safety": 0.14,
            "rule_compliance": 0.14,
            "latency_efficiency": 0.08,
            "reduced_planner_cost": 0.06,
        }
        reward = 0.0
        for key, weight in weights.items():
            reward += float(breakdown.get(key, 0.0)) * weight
        return _clamp(reward), breakdown

    def capture(self, episode: MLTrainingEpisode) -> tuple[MLTrainingEpisode, float, dict[str, float]]:
        reward, breakdown = self.compute_reward(episode)
        enriched = episode.model_copy(update={"outcome": episode.outcome.model_copy(update={"reward": reward})})
        payload = enriched.model_dump(mode="json")
        payload["reward_breakdown"] = breakdown

        with self._lock:
            self._episodes.append(enriched)
            if len(self._episodes) > self.max_in_memory:
                self._episodes = self._episodes[-self.max_in_memory :]
            self._counters["observed_total"] += 1
            source_key = f"observed_{enriched.decision_source.value}"
            self._counters[source_key] = self._counters.get(source_key, 0) + 1

        try:
            self._append_jsonl(payload)
        except Exception:
            logger.exception(
                "ml_observation_persist_failed",
                extra={"event": "ml_observation_persist_failed", "episode_id": enriched.episode_id, "bot_id": enriched.bot_id},
            )

        return enriched, reward, breakdown

    def recent(self, *, bot_id: str | None = None, limit: int = 200) -> list[MLTrainingEpisode]:
        with self._lock:
            rows = list(self._episodes)
        if bot_id:
            rows = [item for item in rows if item.bot_id == bot_id]
        rows = rows[-max(1, int(limit)) :]
        return rows

    def counters(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counters)

